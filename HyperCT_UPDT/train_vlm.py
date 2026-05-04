"""
VLM Training Script for HyperCT_UPDT Pipeline

Trains a Vision-Language Model using precomputed DINOv3 vision tokens:
    1. Load precomputed tokens (.npz) per CT volume
    2. Pass through Q-Former adapter → (num_queries, 4096)
    3. Inject into Llama 3.1 8B at IMAGE_TOKEN positions
    4. Fine-tune LLM with LoRA on VQA pairs

Usage:
    torchrun --nproc_per_node=4 train_vlm.py \
        --tokens_dir ./precompute_tokens_ff \
        --data_json ./train_data.json \
        --val_data_json ./valid_data.json \
        --val_tokens_dir ./precompute_tokens_valid_ff \
        --output_dir ./checkpoints \
        --llm_name meta-llama/Llama-3.1-8B-Instruct
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.qformer import QFormerAdapter
from config import RADIOLOGICAL_TASKS
import os
import json
import argparse
import logging
import inspect
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, PeftModel, get_peft_model


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


TASK_ALIASES = {
    task: sorted({
        task.replace("_", " "),
        task.replace("_", "-"),
        task,
    })
    for task in RADIOLOGICAL_TASKS
}

QUESTION_TYPES = (
    "long_answer",
    "short_answer",
    "multiple_choice",
    "report_generation",
)

QUESTION_TYPE_LABELS = {
    "long_answer": "Long answer",
    "short_answer": "Short answer",
    "multiple_choice": "Multiple choice",
    "report_generation": "Report generation",
}

TYPE_AWARE_INSTRUCTIONS = {
    "long_answer": (
        "Answer as a radiologist using this compact structure: Findings: state "
        "the CT-supported positive findings and relevant negatives asked by the "
        "question. Impression: give the final clinical conclusion. Mention "
        "location, laterality, and severity only when supported. Do not invent "
        "uncertain findings."
    ),
    "short_answer": (
        "Answer concisely in one short phrase or sentence. If the question is "
        "polar, answer Yes or No first. Do not add unsupported findings."
    ),
    "multiple_choice": (
        "Choose the single best option. Start the answer with only the option "
        "letter, then give at most one short justification."
    ),
    "report_generation": (
        "Generate a concise radiology-style answer using exactly two sections: "
        "Findings and Impression. Use only CT-supported findings. Do not invent "
        "abnormalities, locations, comparisons, or severity."
    ),
}
TASK_ALIASES.update({
    "medical_material": ["medical material", "medical device", "device", "catheter", "tube", "line"],
    "arterial_wall_calcification": ["arterial wall calcification", "vascular calcification", "aortic calcification"],
    "coronary_artery_wall_calcification": ["coronary artery wall calcification", "coronary calcification", "coronary artery calcification"],
    "pericardial_effusion": ["pericardial effusion"],
    "pleural_effusion": ["pleural effusion"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "lymphadenopathy": ["lymphadenopathy", "enlarged lymph node", "enlarged lymph nodes"],
    "emphysema": ["emphysema", "emphysematous change", "emphysematous changes"],
    "atelectasis": ["atelectasis", "atelectatic change", "atelectatic changes"],
    "nodule": ["nodule", "nodules", "pulmonary nodule", "pulmonary nodules"],
    "opacity": ["opacity", "opacities", "ground glass opacity", "ground-glass opacity"],
    "fibrosis": ["fibrosis", "fibrotic change", "fibrotic changes"],
    "mosaic_attenuation": ["mosaic attenuation"],
    "peribronchial_thickening": ["peribronchial thickening", "bronchial wall thickening"],
    "consolidation": ["consolidation", "airspace consolidation"],
    "bronchiectasis": ["bronchiectasis", "bronchiectatic change", "bronchiectatic changes"],
    "interlobular_septal_thickening": ["interlobular septal thickening", "septal thickening"],
})


def normalize_answer(text: str) -> str:
    """Normalize generated/reference answers for exact-match style metrics."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_tokens(text: str) -> List[str]:
    return normalize_answer(text).split()


def ngram_counts(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu1_score(prediction: str, reference: str) -> float:
    pred_tokens = metric_tokens(prediction)
    ref_tokens = metric_tokens(reference)
    if not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    clipped = sum((pred_counts & ref_counts).values())
    precision = clipped / len(pred_tokens)
    if not ref_tokens:
        return 0.0
    brevity_penalty = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(
        1.0 - len(ref_tokens) / max(len(pred_tokens), 1)
    )
    return brevity_penalty * precision


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = metric_tokens(prediction)
    ref_tokens = metric_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def meteor_score(prediction: str, reference: str) -> float:
    pred_tokens = metric_tokens(prediction)
    ref_tokens = metric_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_available = defaultdict(list)
    for idx, token in enumerate(ref_tokens):
        ref_available[token].append(idx)

    matches = []
    used_ref = set()
    for pred_idx, token in enumerate(pred_tokens):
        for ref_idx in ref_available.get(token, []):
            if ref_idx not in used_ref:
                used_ref.add(ref_idx)
                matches.append((pred_idx, ref_idx))
                break

    match_count = len(matches)
    if match_count == 0:
        return 0.0

    precision = match_count / len(pred_tokens)
    recall = match_count / len(ref_tokens)
    fmean = (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) else 0.0

    matches.sort()
    chunks = 1
    for prev, cur in zip(matches, matches[1:]):
        if cur[0] != prev[0] + 1 or cur[1] != prev[1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / match_count) ** 3
    return fmean * (1.0 - penalty)


def cider_scores(predictions: List[str], references: List[str]) -> List[float]:
    """Lightweight CIDEr-style TF-IDF cosine score over 1-4 grams."""
    if not predictions:
        return []

    tokenized_refs = [metric_tokens(ref) for ref in references]
    n_doc = max(len(tokenized_refs), 1)
    doc_freq = {n: Counter() for n in range(1, 5)}
    for tokens in tokenized_refs:
        for n in range(1, 5):
            doc_freq[n].update(set(ngram_counts(tokens, n).keys()))

    def tfidf_vector(tokens: List[str], n: int) -> Dict[Tuple[str, ...], float]:
        counts = ngram_counts(tokens, n)
        total = sum(counts.values())
        if total == 0:
            return {}
        vec = {}
        for gram, count in counts.items():
            tf = count / total
            idf = math.log((n_doc + 1.0) / (doc_freq[n].get(gram, 0) + 1.0))
            vec[gram] = tf * idf
        return vec

    def cosine(vec_a: Dict[Tuple[str, ...], float], vec_b: Dict[Tuple[str, ...], float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = metric_tokens(pred)
        ref_tokens = metric_tokens(ref)
        score = 0.0
        for n in range(1, 5):
            score += cosine(tfidf_vector(pred_tokens, n), tfidf_vector(ref_tokens, n))
        scores.append(10.0 * score / 4.0)
    return scores


def classify_yes_no(text: str) -> Optional[int]:
    """Return 1 for yes/present, 0 for no/absent, None when unclear."""
    norm = normalize_answer(text)
    if not norm:
        return None

    if re.match(r"^(yes|yeah|yep|present|positive)\b", norm):
        return 1
    if re.match(r"^(no|nope|negative|absent|none)\b", norm):
        return 0

    negative_patterns = [
        r"\bno evidence of\b", r"\bwithout\b", r"\babsent\b",
        r"\bnegative for\b", r"\bfree of\b", r"\bnot seen\b",
        r"\bnot present\b", r"\bno\b",
    ]
    positive_patterns = [
        r"\bpresent\b", r"\bpositive for\b", r"\bevidence of\b",
        r"\bseen\b", r"\bvisualized\b", r"\bthere is\b",
        r"\bthere are\b", r"\bshows\b", r"\bdemonstrates\b",
    ]
    if any(re.search(pattern, norm) for pattern in negative_patterns):
        return 0
    if any(re.search(pattern, norm) for pattern in positive_patterns):
        return 1
    return None


def extract_mcq_options(question: str) -> Dict[str, str]:
    """Parse option letters and option text from an MCQ prompt when available."""
    option_pattern = re.compile(
        r"(?:^|\n|\s)([A-Da-d])[\)\.:]\s*(.*?)(?=(?:\n|\s)[A-Da-d][\)\.:]\s*|$)",
        re.DOTALL,
    )
    options = {}
    for match in option_pattern.finditer(str(question)):
        letter = match.group(1).upper()
        option_text = " ".join(match.group(2).strip().split())
        if option_text:
            options[letter] = option_text
    return options


def extract_mcq_option(text: str, question: Optional[str] = None) -> Optional[str]:
    """Extract an A/B/C/D choice from raw generated/reference text."""
    raw = str(text).strip()
    if not raw:
        return None
    cleaned = clean_generated_answer(raw)
    candidates = [raw, cleaned]
    patterns = (
        r"^\s*([A-Da-d])\s*$",
        r"^\s*([A-Da-d])[\)\.:]\s+",
        r"^\s*(?:answer|option|choice)\s*(?:is|:|-)?\s*([A-Da-d])\b",
        r"\b(?:answer|option|choice)\s*(?:is|:|-)?\s*([A-Da-d])\b",
        r"\(([A-Da-d])\)",
    )
    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if match:
                return match.group(1).upper()

    if question:
        answer_norm = normalize_answer(cleaned)
        for letter, option_text in extract_mcq_options(question).items():
            option_norm = normalize_answer(option_text)
            if len(option_norm) >= 3 and (
                answer_norm == option_norm
                or answer_norm in option_norm
                or option_norm in answer_norm
            ):
                return letter
    return None


def infer_task_from_text(text: str) -> Optional[str]:
    norm = normalize_answer(text)
    for task, aliases in TASK_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_answer(alias)
            if alias_norm and re.search(rf"\b{re.escape(alias_norm)}\b", norm):
                return task
    return None


def _is_negated_near(text: str, start: int) -> bool:
    window = normalize_answer(text[max(0, start - 90):start])
    return bool(re.search(
        r"\b(no|without|absent|negative|free of|lack of|lacks|not|denies)\b",
        window,
    ))


def extract_clinical_labels(text: str) -> Dict[str, int]:
    """Extract task-level labels from short generated/reference answers."""
    labels: Dict[str, int] = {}
    norm = normalize_answer(text)
    for task, aliases in TASK_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_answer(alias)
            if not alias_norm:
                continue
            match = re.search(rf"\b{re.escape(alias_norm)}\b", norm)
            if match:
                labels[task] = 0 if _is_negated_near(norm, match.start()) else 1
                break
    return labels


def normalize_question_type(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = normalize_answer(str(raw))
    if not text:
        return None
    if "multiple" in text or "mcq" in text or "choice" in text:
        return "multiple_choice"
    if "report" in text or "impression" in text:
        return "report_generation"
    if "short" in text or "yes no" in text or "yes/no" in str(raw).lower():
        return "short_answer"
    if "long" in text or "conversation" in text or "description" in text or "free response" in text:
        return "long_answer"
    return None


def clean_question_text(question: str) -> str:
    return str(question).replace(IMAGE_TOKEN, " ").strip()


def infer_question_type(item: dict, question: str) -> str:
    """Infer CT-CHAT-style question type without inspecting the reference answer."""
    metadata_keys = (
        "question_type", "type", "category", "task_type", "answer_type",
        "qa_type", "source", "subset",
    )
    for key in metadata_keys:
        qtype = normalize_question_type(item.get(key))
        if qtype:
            return qtype

    for container_key in ("metadata", "meta"):
        metadata = item.get(container_key)
        if isinstance(metadata, dict):
            for key in metadata_keys:
                qtype = normalize_question_type(metadata.get(key))
                if qtype:
                    return qtype

    clean_question = clean_question_text(question)
    q = clean_question.lower()
    q_norm = normalize_answer(clean_question)
    if re.search(r"\b(a|b|c|d)[\)\.:]\s", q) or "which of the following" in q or "options:" in q:
        return "multiple_choice"
    if "report" in q or "impression" in q or "findings section" in q:
        return "report_generation"
    if re.match(
        r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|should)\b",
        q,
    ) or re.search(r"\b(is there|are there|present or absent|yes or no)\b", q):
        return "short_answer"
    if len(q_norm.split()) <= 12:
        return "short_answer"
    return "long_answer"


def is_yes_no_question(question: str) -> bool:
    q = clean_question_text(question).lower()
    return bool(
        re.match(
            r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|should)\b",
            q,
        )
        or re.search(r"\b(is there|are there|present or absent|yes or no)\b", q)
    )


def add_type_instruction(content: str, question_type: str) -> str:
    if question_type == "short_answer" and is_yes_no_question(content):
        instruction = (
            "Answer with exactly Yes or No. Do not add unsupported findings or "
            "extra explanation."
        )
    else:
        instruction = TYPE_AWARE_INSTRUCTIONS.get(question_type)
    if not instruction:
        return content
    if "Answering constraint:" in content:
        return content
    return f"{content}\n\nAnswering constraint: {instruction}"


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def format_task_hint(task_scores: List[Tuple[str, float]]) -> str:
    if not task_scores:
        return ""
    readable = ", ".join(
        f"{task.replace('_', ' ')} ({prob:.2f})"
        for task, prob in task_scores
    )
    return (
        "Stage 1 visual finding hints from the CT classifier, with estimated "
        f"presence probabilities: {readable}. Use these only as visual guidance; "
        "the final answer must still match the question and avoid unsupported "
        "findings."
    )


def add_task_hint(content: str, hint: str) -> str:
    if not hint or "Stage 1 visual finding hints" in content:
        return content
    return f"{content}\n\n{hint}"


def group_indices_by_type(records: List[dict]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {qtype: [] for qtype in QUESTION_TYPES}
    for idx, record in enumerate(records):
        qtype = record.get("question_type", "long_answer")
        groups.setdefault(qtype, []).append(idx)
    return groups


def add_type_metrics(metrics: Dict[str, float], records: List[dict],
                     predictions: List[str], references: List[str]) -> None:
    for qtype, indices in group_indices_by_type(records).items():
        if not indices:
            continue
        pred_subset = [predictions[i] for i in indices]
        ref_subset = [references[i] for i in indices]
        prefix = f"generation_{qtype}"
        exact = sum(
            int(normalize_answer(pred) == normalize_answer(ref))
            for pred, ref in zip(pred_subset, ref_subset)
        )
        token_f1_values = [
            token_f1(pred, ref) for pred, ref in zip(pred_subset, ref_subset)
        ]
        bleu_values = [
            bleu1_score(pred, ref) for pred, ref in zip(pred_subset, ref_subset)
        ]
        meteor_values = [
            meteor_score(pred, ref) for pred, ref in zip(pred_subset, ref_subset)
        ]
        rouge_values = [
            rouge_l_score(pred, ref) for pred, ref in zip(pred_subset, ref_subset)
        ]
        cider_values = cider_scores(pred_subset, ref_subset)
        metrics[f"{prefix}_samples"] = len(indices)
        metrics[f"{prefix}_exact_match"] = exact / len(indices)
        metrics[f"{prefix}_token_f1"] = sum(token_f1_values) / len(token_f1_values)
        metrics[f"{prefix}_bleu1"] = sum(bleu_values) / len(bleu_values)
        metrics[f"{prefix}_meteor"] = sum(meteor_values) / len(meteor_values)
        metrics[f"{prefix}_rouge_l"] = sum(rouge_values) / len(rouge_values)
        metrics[f"{prefix}_cider"] = sum(cider_values) / len(cider_values) if cider_values else 0.0


def binary_scores(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class VQADataset(Dataset):
    """
    Dataset loading precomputed vision tokens + VQA conversations.

    Expected data_json format (list of dicts):
        {
            "id": "scan_001",
            "image": "scan_001.npz",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe findings."},
                {"from": "gpt", "value": "The scan shows..."}
            ]
        }
    """

    def __init__(self, data_json: str, tokens_dir: str, tokenizer,
                 max_length: int = 2048, num_task_tokens: int = 3,
                 type_aware_prompts: bool = False,
                 task_hint_top_k: int = 0,
                 balance_question_types: bool = False,
                 long_answer_oversample_factor: int = 1,
                 report_generation_oversample_factor: int = 1):
        with open(data_json, "r") as f:
            raw_data = json.load(f)
        self.tokens_dir = tokens_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_task_tokens = num_task_tokens
        self.type_aware_prompts = type_aware_prompts
        self.task_hint_top_k = max(0, task_hint_top_k)
        self.balance_question_types = balance_question_types
        self.long_answer_oversample_factor = max(1, long_answer_oversample_factor)
        self.report_generation_oversample_factor = max(1, report_generation_oversample_factor)
        self._token_stems = self._discover_token_stems(tokens_dir)
        self._token_path_cache: Dict[str, str] = {}

        self.data = []
        missing_image = 0
        missing_tokens = 0
        missing_conversations = 0
        for item in raw_data:
            if not isinstance(item, dict):
                missing_image += 1
                continue
            convs = item.get("conversations")
            if not isinstance(convs, list) or not convs:
                missing_conversations += 1
                continue

            image_ref = self._extract_image_ref(item)
            if image_ref is None:
                missing_image += 1
                continue
            if not self._has_token_file(image_ref):
                missing_tokens += 1
                continue

            normalized = dict(item)
            normalized["_image_ref"] = image_ref
            normalized["_question_type"] = self._infer_item_question_type(normalized)
            self.data.append(normalized)

        log.info(
            f"Loaded {len(self.data)}/{len(raw_data)} VQA samples from {data_json} "
            f"with tokens in {tokens_dir}"
        )
        if missing_image or missing_tokens or missing_conversations:
            log.warning(
                f"Filtered VQA samples: missing_image={missing_image}, "
                f"missing_tokens={missing_tokens}, "
                f"missing_conversations={missing_conversations}"
            )
        if not self.data:
            raise ValueError(
                f"No usable VQA samples found in {data_json}. Check image keys "
                f"and token coverage in {tokens_dir}."
            )

        self._log_question_type_counts("Question type counts before balancing")
        if self.balance_question_types:
            self._balance_question_types()
            self._log_question_type_counts("Question type counts after balancing")
        if self.long_answer_oversample_factor > 1 or self.report_generation_oversample_factor > 1:
            self._oversample_long_report_types()
            self._log_question_type_counts("Question type counts after long/report oversampling")

        # Llama 3.1 uses <|eot_id|> for end-of-turn, distinct from eos_token_id
        self.eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.eot_token_id is None or self.eot_token_id == tokenizer.unk_token_id:
            self.eot_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _first_user_question(item: dict) -> str:
        for conv in item.get("conversations", []):
            role = conv.get("from", conv.get("role", ""))
            if role in {"human", "user"}:
                return conv.get("value", conv.get("content", ""))
        return ""

    def _infer_item_question_type(self, item: dict) -> str:
        return infer_question_type(item, self._first_user_question(item))

    def _log_question_type_counts(self, message: str) -> None:
        counts = Counter(item.get("_question_type", "long_answer") for item in self.data)
        log.info("%s: %s", message, dict(sorted(counts.items())))

    def _balance_question_types(self) -> None:
        groups: Dict[str, List[dict]] = defaultdict(list)
        for item in self.data:
            groups[item.get("_question_type", "long_answer")].append(item)
        nonempty = {qtype: items for qtype, items in groups.items() if items}
        if len(nonempty) <= 1:
            return

        target = max(len(items) for items in nonempty.values())
        balanced = []
        for qtype in QUESTION_TYPES:
            items = nonempty.get(qtype, [])
            if not items:
                continue
            repeats, remainder = divmod(target, len(items))
            balanced.extend(items * repeats)
            balanced.extend(items[:remainder])

        # Preserve any unexpected types without dropping data.
        for qtype, items in nonempty.items():
            if qtype not in QUESTION_TYPES:
                balanced.extend(items)
        self.data = balanced

    def _oversample_long_report_types(self) -> None:
        factors = {
            "long_answer": self.long_answer_oversample_factor,
            "report_generation": self.report_generation_oversample_factor,
        }
        expanded = []
        for item in self.data:
            qtype = item.get("_question_type", "long_answer")
            expanded.extend([item] * factors.get(qtype, 1))
        self.data = expanded

    @staticmethod
    def _discover_token_stems(tokens_dir: str) -> set:
        if not os.path.isdir(tokens_dir):
            return set()
        return {
            os.path.splitext(name)[0]
            for name in os.listdir(tokens_dir)
            if name.endswith(".npz")
        }

    @staticmethod
    def _token_stem_from_ref(image_ref: str) -> str:
        base = os.path.basename(str(image_ref).strip())
        if base.endswith(".nii.gz"):
            return base[:-7]
        return os.path.splitext(base)[0]

    @staticmethod
    def _find_volume_like_string(value) -> Optional[str]:
        if isinstance(value, str):
            match = re.search(
                r"(?:train|valid|validation|test)_\d+_[A-Za-z]_\d+"
                r"(?:\.nii\.gz|\.npz)?",
                value,
            )
            if match:
                return match.group(0)
            if value.endswith((".nii.gz", ".npz")):
                return value
        elif isinstance(value, list):
            for entry in value:
                found = VQADataset._find_volume_like_string(entry)
                if found:
                    return found
        elif isinstance(value, dict):
            for entry in value.values():
                found = VQADataset._find_volume_like_string(entry)
                if found:
                    return found
        return None

    @staticmethod
    def _extract_image_ref(item: dict) -> Optional[str]:
        image_keys = (
            "image", "images", "image_path", "image_name", "filename",
            "file_name", "volume", "volume_name", "VolumeName", "path",
            "scan", "scan_id", "ct", "ct_path", "nii", "nii_path",
        )
        for key in image_keys:
            if key in item:
                found = VQADataset._find_volume_like_string(item[key])
                if found:
                    return found

        for key in ("id", "uid", "study_id", "sample_id"):
            if key in item:
                found = VQADataset._find_volume_like_string(item[key])
                if found:
                    return found

        # Last resort: scan all string/list/dict values. Some VQA exports store
        # the volume name only inside metadata or IDs rather than an image field.
        return VQADataset._find_volume_like_string(item)

    def _has_token_file(self, image_ref: str) -> bool:
        return self._token_stem_from_ref(image_ref) in self._token_stems

    def _resolve_token_path(self, image_ref: str) -> str:
        cached = self._token_path_cache.get(image_ref)
        if cached is not None:
            return cached

        candidates = []
        candidates.append(os.path.join(self.tokens_dir, image_ref))
        stem = self._token_stem_from_ref(image_ref)
        candidates.append(os.path.join(self.tokens_dir, f"{stem}.npz"))

        for path in candidates:
            if os.path.exists(path):
                self._token_path_cache[image_ref] = path
                return path
        raise FileNotFoundError(
            f"Could not find precomputed tokens for image={image_ref!r}. "
            f"Tried: {candidates}"
        )

    def _load_vision_tokens(self, item: dict) -> torch.Tensor:
        image_ref = item["_image_ref"]
        npz_path = self._resolve_token_path(image_ref)
        npz_data = np.load(npz_path)
        all_tokens = npz_data["tokens"]       # (num_tasks, T_out, 768)
        predictions = npz_data["predictions"]  # (num_tasks, num_tasks)

        # Select top-k tasks by classifier confidence (diagonal logits)
        # Diagonal[i] = how well task i's LoRA detects task i in this volume
        diag = np.array(np.diag(predictions), copy=True)

        # Soft weighted combination: instead of hard top-k selection (which
        # can propagate errors if the classifier picks wrong tasks), use
        # softmax-weighted sum across ALL tasks. Low-confidence tasks
        # contribute minimally; no hard thresholding risk.
        weights = torch.softmax(torch.from_numpy(diag).float(), dim=0)
        all_tokens_t = torch.from_numpy(np.array(
            all_tokens, copy=True)).float()  # (num_tasks, T_out, 768)
        # Weighted sum: (T_out, 768) — each task's tokens weighted by confidence
        return torch.einsum('t, t n d -> n d', weights, all_tokens_t)

    def _stage1_task_hint(self, item: dict) -> str:
        if self.task_hint_top_k <= 0:
            return ""
        npz_path = self._resolve_token_path(item["_image_ref"])
        npz_data = np.load(npz_path)
        predictions = npz_data["predictions"]
        diag = np.array(np.diag(predictions), copy=True)
        probs = sigmoid_np(diag)
        top_k = min(self.task_hint_top_k, len(RADIOLOGICAL_TASKS))
        top_indices = np.argsort(-probs)[:top_k]
        task_scores = [
            (RADIOLOGICAL_TASKS[int(idx)], float(probs[int(idx)]))
            for idx in top_indices
        ]
        return format_task_hint(task_scores)

    def _format_training_text(self, item: dict) -> str:
        convs = item["conversations"]
        text_parts = []
        task_hint = self._stage1_task_hint(item)
        has_image_token = any(
            IMAGE_TOKEN in str(conv.get("value", conv.get("content", "")))
            for conv in convs
        )
        inserted_image_token = has_image_token
        for conv in convs:
            role = conv.get("from", conv.get("role", ""))
            content = conv.get("value", conv.get("content", ""))
            if role in {"human", "user"}:
                if not inserted_image_token:
                    content = f"{IMAGE_TOKEN}\n{content}"
                    inserted_image_token = True
                if self.type_aware_prompts:
                    qtype = item.get("_question_type") or infer_question_type(item, content)
                    content = add_type_instruction(content, qtype)
                content = add_task_hint(content, task_hint)
                text_parts.append(
                    f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role in {"gpt", "assistant"}:
                text_parts.append(
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")

        return "<|begin_of_text|>" + "".join(text_parts)

    def _format_generation_prompt(self, item: dict) -> Tuple[str, str, str, str]:
        """Return prompt, reference answer, last user question, and question type."""
        convs = item["conversations"]
        text_parts = []
        last_question = ""
        question_type = item.get("_question_type", "long_answer")
        task_hint = self._stage1_task_hint(item)
        has_image_token = any(
            IMAGE_TOKEN in str(conv.get("value", conv.get("content", "")))
            for conv in convs
        )
        inserted_image_token = has_image_token
        for conv in convs:
            role = conv.get("from", conv.get("role", ""))
            content = conv.get("value", conv.get("content", ""))
            if role in {"human", "user"}:
                if not inserted_image_token:
                    content = f"{IMAGE_TOKEN}\n{content}"
                    inserted_image_token = True
                last_question = content
                question_type = item.get("_question_type") or infer_question_type(item, content)
                if self.type_aware_prompts:
                    content = add_type_instruction(content, question_type)
                content = add_task_hint(content, task_hint)
                text_parts.append(
                    f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role in {"gpt", "assistant"}:
                prompt = (
                    "<|begin_of_text|>"
                    + "".join(text_parts)
                    + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                return prompt, content, last_question, question_type
        raise ValueError("No assistant answer found in VQA conversation")

    def _tokenize_with_image(self, full_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tokenize — split around IMAGE_TOKEN
        chunks = full_text.split(IMAGE_TOKEN)
        input_ids = []
        image_token_positions = []

        for i, chunk in enumerate(chunks):
            if chunk:
                chunk_ids = self.tokenizer.encode(
                    chunk, add_special_tokens=False)
                input_ids.extend(chunk_ids)
            if i < len(chunks) - 1:
                image_token_positions.append(len(input_ids))
                input_ids.append(IMAGE_TOKEN_INDEX)

        input_ids = torch.tensor(input_ids[:self.max_length], dtype=torch.long)

        # Filter positions that survived truncation
        image_token_positions = [
            p for p in image_token_positions if p < len(input_ids)]
        return input_ids, torch.tensor(image_token_positions, dtype=torch.long)

    def build_generation_sample(self, idx: int) -> dict:
        item = self.data[idx]
        prompt, reference, question, question_type = self._format_generation_prompt(
            item)
        input_ids, image_positions = self._tokenize_with_image(prompt)
        return {
            "id": item.get("id", str(idx)),
            "image": item["_image_ref"],
            "question": question,
            "question_type": question_type,
            "question_type_label": QUESTION_TYPE_LABELS.get(question_type, question_type),
            "is_yes_no_question": is_yes_no_question(question),
            "reference": reference,
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "vision_tokens": self._load_vision_tokens(item),
            "image_positions": image_positions,
        }

    def __getitem__(self, idx):
        item = self.data[idx]

        vision_tokens = self._load_vision_tokens(item)
        full_text = self._format_training_text(item)
        input_ids, image_positions = self._tokenize_with_image(full_text)

        # Labels: mask everything before assistant responses
        labels = input_ids.clone()
        # Mask image tokens
        labels[input_ids == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        # Mask user turns (everything before assistant header)
        assistant_header = self.tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
        )
        header_len = len(assistant_header)

        # Simple masking: mask everything that isn't in an assistant turn.
        # The assistant's <|eot_id|> IS included in the loss so the model
        # learns to produce the end-of-turn signal.
        in_assistant = False
        for i in range(len(labels)):
            if i + header_len <= len(input_ids):
                if input_ids[i:i+header_len].tolist() == assistant_header:
                    in_assistant = True
            if not in_assistant:
                labels[i] = IGNORE_INDEX
            # Llama 3.1: <|eot_id|> marks end-of-turn (NOT eos_token_id).
            # Check AFTER the masking decision so eot_id in assistant
            # turns is supervised (model learns when to stop).
            if input_ids[i].item() == self.eot_token_id:
                in_assistant = False

        return {
            "input_ids": input_ids,
            "labels": labels,
            "vision_tokens": vision_tokens,
            "image_positions": image_positions,
        }


GENERATION_EVAL_BUCKETS = (
    "long_answer",
    "short_answer",
    "yes_no",
    "multiple_choice",
    "report_generation",
)


def generation_eval_bucket(item: dict) -> str:
    qtype = item.get("_question_type", "long_answer")
    question = VQADataset._first_user_question(item)
    if qtype == "short_answer" and is_yes_no_question(question):
        return "yes_no"
    return qtype if qtype in QUESTION_TYPES else "long_answer"


def select_generation_eval_indices(
    dataset: VQADataset,
    max_samples: int,
    stratified: bool,
    samples_per_bucket: int,
) -> List[int]:
    if max_samples < 0:
        return list(range(len(dataset)))

    sample_count = min(max_samples, len(dataset))
    if not stratified:
        return list(range(sample_count))

    groups: Dict[str, List[int]] = {bucket: [] for bucket in GENERATION_EVAL_BUCKETS}
    for idx, item in enumerate(dataset.data):
        bucket = generation_eval_bucket(item)
        groups.setdefault(bucket, []).append(idx)

    target = samples_per_bucket if samples_per_bucket > 0 else max(
        1, math.ceil(sample_count / len(GENERATION_EVAL_BUCKETS))
    )
    selected = []
    used = set()
    for bucket in GENERATION_EVAL_BUCKETS:
        for idx in groups.get(bucket, [])[:target]:
            if idx not in used:
                selected.append(idx)
                used.add(idx)

    if len(selected) < sample_count:
        for idx in range(len(dataset)):
            if idx not in used:
                selected.append(idx)
                used.add(idx)
            if len(selected) >= sample_count:
                break

    return selected[:sample_count]


def collate_fn(batch, pad_token_id: int):
    """Pad sequences and stack vision tokens."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = []
    labels = []
    attention_mask = []
    vision_tokens = []
    image_positions = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(
            F.pad(b["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(F.pad(b["labels"], (0, pad_len), value=IGNORE_INDEX))
        mask = torch.ones(seq_len, dtype=torch.long)
        attention_mask.append(F.pad(mask, (0, pad_len), value=0))
        vision_tokens.append(b["vision_tokens"])
        image_positions.append(b["image_positions"])

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
        "vision_tokens": vision_tokens,  # list of variable-size tensors
        "image_positions": image_positions,
    }


class HyperCTVLM(nn.Module):
    """
    Full VLM: Q-Former adapter + LLM.

    Forward:
        1. Q-Former: vision_tokens (B, N, 768) → (B, num_queries, 4096)
        2. Replace IMAGE_TOKEN positions in input embeddings with vision features
        3. LLM forward with modified embeddings
    """

    def __init__(self, llm, qformer: QFormerAdapter):
        super().__init__()
        self.llm = llm
        self.qformer = qformer

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate to LLM so HuggingFace Trainer can enable gradient checkpointing."""
        self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def _build_multimodal_inputs(
        self,
        input_ids,
        attention_mask,
        vision_tokens,
        image_positions,
        labels=None,
    ):
        device = input_ids.device
        B = input_ids.shape[0]

        # Get text embeddings (clamp IMAGE_TOKEN_INDEX to valid range)
        text_embeds = self.llm.get_input_embeddings()(
            input_ids.clamp(min=0)
        )  # (B, seq_len, hidden_dim)

        new_embeds_list = []
        new_labels_list = []
        new_mask_list = []

        for i in range(B):
            if len(image_positions[i]) > 0:
                # Process vision tokens through Q-Former
                v_tokens = vision_tokens[i].unsqueeze(
                    0).to(device)  # (1, N, 768)
                aligned = self.qformer(v_tokens)  # (1, num_queries, 4096)
                aligned = aligned.squeeze(0).to(
                    text_embeds.dtype)  # (num_queries, 4096)
                num_vision = aligned.shape[0]

                # Replace IMAGE_TOKEN placeholder with vision features
                pos = image_positions[i][0].item()
                pre_emb = text_embeds[i, :pos]
                post_emb = text_embeds[i, pos + 1:]
                sample_embeds = torch.cat([pre_emb, aligned, post_emb], dim=0)

                if labels is not None:
                    # Adjust labels around the inserted vision features.
                    pre_lab = labels[i, :pos]
                    post_lab = labels[i, pos + 1:]
                    vis_lab = torch.full(
                        (num_vision,), IGNORE_INDEX, device=device, dtype=labels.dtype)
                    sample_labels = torch.cat([pre_lab, vis_lab, post_lab])
                else:
                    sample_labels = None

                pre_mask = attention_mask[i, :pos]
                post_mask = attention_mask[i, pos + 1:]
                vis_mask = torch.ones(
                    num_vision, device=device, dtype=attention_mask.dtype)
                sample_mask = torch.cat([pre_mask, vis_mask, post_mask])
            else:
                sample_embeds = text_embeds[i]
                sample_labels = labels[i] if labels is not None else None
                sample_mask = attention_mask[i]

            new_embeds_list.append(sample_embeds)
            if sample_labels is not None:
                new_labels_list.append(sample_labels)
            new_mask_list.append(sample_mask)

        # Pad all samples to the same max length after vision injection
        max_len = max(e.shape[0] for e in new_embeds_list)
        hidden_dim = text_embeds.shape[-1]

        padded_embeds = torch.zeros(
            B, max_len, hidden_dim, device=device, dtype=text_embeds.dtype)
        padded_labels = None
        if labels is not None:
            padded_labels = torch.full(
                (B, max_len), IGNORE_INDEX, device=device, dtype=labels.dtype)
        padded_mask = torch.zeros(
            B, max_len, device=device, dtype=attention_mask.dtype)

        for i in range(B):
            cur_len = new_embeds_list[i].shape[0]
            padded_embeds[i, :cur_len] = new_embeds_list[i]
            if padded_labels is not None:
                padded_labels[i, :cur_len] = new_labels_list[i]
            padded_mask[i, :cur_len] = new_mask_list[i]

        return padded_embeds, padded_mask, padded_labels

    def forward(self, input_ids, labels, attention_mask, vision_tokens, image_positions):
        padded_embeds, padded_mask, padded_labels = self._build_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_tokens=vision_tokens,
            image_positions=image_positions,
            labels=labels,
        )

        outputs = self.llm(
            inputs_embeds=padded_embeds,
            labels=padded_labels,
            attention_mask=padded_mask,
        )
        return outputs

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, vision_tokens, image_positions, **kwargs):
        padded_embeds, padded_mask, _ = self._build_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_tokens=vision_tokens,
            image_positions=image_positions,
            labels=None,
        )
        return self.llm.generate(
            inputs_embeds=padded_embeds,
            attention_mask=padded_mask,
            **kwargs,
        )


def find_linear_names(model):
    """Find all linear layer names for LoRA, excluding vision/projector modules."""
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            # Skip embeddings and head
            if any(k in name for k in ["embed", "lm_head", "qformer"]):
                continue
            names.add(parts[-1])
    return sorted(names)


class PerplexityTrainer(Trainer):
    """Trainer that adds eval perplexity next to eval loss."""

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        prefix = kwargs.get("metric_key_prefix", "eval")
        loss_key = f"{prefix}_loss"
        if loss_key in metrics:
            loss = metrics[loss_key]
            perplexity = math.exp(loss) if math.isfinite(loss) and loss < 20 else float("inf")
            ppl_key = f"{prefix}_perplexity"
            metrics[ppl_key] = perplexity
            self.log({ppl_key: perplexity})
        return metrics


def clean_generated_answer(text: str) -> str:
    """Strip chat-template control tokens from generated text."""
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    if "<|eot_id|>" in text:
        text = text.split("<|eot_id|>")[0]
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    return " ".join(text.replace("assistant", " ").split())


def extract_first_number(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def extract_json_object(text: str) -> Optional[dict]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_llama_score_prompt(reference: str, candidate: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a strict radiology report evaluator. Score the candidate answer "
        "against the reference answer for clinical correctness. Focus on factual "
        "medical agreement, not wording. Return only JSON with keys score and explanation. "
        "score must be a number from 0 to 10, where 10 is clinically identical."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        "Return JSON only."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_green_prompt(reference: str, candidate: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Objective: Evaluate the accuracy of a candidate radiology report in comparison "
        "to a reference radiology report composed by expert radiologists. Count clinically "
        "significant and clinically insignificant errors. Errors include false findings, "
        "missing findings, wrong anatomic location, wrong severity, inappropriate comparison, "
        "or omitted comparison. Focus on clinical findings, not wording. Return only JSON "
        "with keys significant_errors, insignificant_errors, green_score, and explanation. "
        "green_score must be from 0 to 1, where 1 means no clinically meaningful error."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Reference report:\n{reference}\n\n"
        f"Candidate report:\n{candidate}\n\n"
        "Return JSON only."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@torch.no_grad()
def generate_text_with_llm(llm, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(llm.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_token_id is None or eot_token_id == tokenizer.unk_token_id:
        eot_token_id = tokenizer.eos_token_id
    generated = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=eot_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    return clean_generated_answer(tokenizer.decode(new_tokens, skip_special_tokens=False))


def score_with_llama_judge(
    llm,
    tokenizer,
    records: List[dict],
    max_samples: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    if max_samples == 0:
        return {}
    sample_count = len(records) if max_samples < 0 else min(max_samples, len(records))
    if sample_count == 0:
        return {}

    scores = []
    type_scores = defaultdict(list)
    for idx, record in enumerate(records[:sample_count]):
        prompt = build_llama_score_prompt(record["reference"], record["prediction"])
        answer = generate_text_with_llm(llm, tokenizer, prompt, max_new_tokens)
        parsed = extract_json_object(answer)
        score = None
        if parsed is not None:
            raw_score = parsed.get("score")
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
        if score is None:
            score = extract_first_number(answer)
        if score is not None:
            score = max(0.0, min(10.0, score))
            scores.append(score)
            type_scores[record.get("question_type", "long_answer")].append(score)
            record["llama_score"] = score
        record["llama_score_raw"] = answer
        if (idx + 1) % 25 == 0:
            log.info(f"Llama judge: {idx + 1}/{sample_count} samples")

    metrics = {
        "generation_llama_score_count": len(scores),
        "generation_llama_score_mean": sum(scores) / len(scores) if scores else 0.0,
    }
    for qtype, values in type_scores.items():
        if values:
            metrics[f"generation_{qtype}_llama_score_count"] = len(values)
            metrics[f"generation_{qtype}_llama_score_mean"] = sum(values) / len(values)
    return metrics


def score_with_green_style_judge(
    llm,
    tokenizer,
    records: List[dict],
    max_samples: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    if max_samples == 0:
        return {}
    sample_count = len(records) if max_samples < 0 else min(max_samples, len(records))
    if sample_count == 0:
        return {}

    green_scores = []
    sig_errors = []
    insig_errors = []
    type_green_scores = defaultdict(list)
    type_sig_errors = defaultdict(list)
    type_insig_errors = defaultdict(list)
    for idx, record in enumerate(records[:sample_count]):
        qtype = record.get("question_type", "long_answer")
        prompt = build_green_prompt(record["reference"], record["prediction"])
        answer = generate_text_with_llm(llm, tokenizer, prompt, max_new_tokens)
        parsed = extract_json_object(answer)

        sig = None
        insig = None
        score = None
        if parsed is not None:
            if isinstance(parsed.get("significant_errors"), (int, float)):
                sig = float(parsed["significant_errors"])
            if isinstance(parsed.get("insignificant_errors"), (int, float)):
                insig = float(parsed["insignificant_errors"])
            if isinstance(parsed.get("green_score"), (int, float)):
                score = float(parsed["green_score"])

        if sig is None or insig is None:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
            if sig is None and numbers:
                sig = float(numbers[0])
            if insig is None and len(numbers) > 1:
                insig = float(numbers[1])
        if score is None and sig is not None:
            # GREEN is error-count based; this bounded proxy keeps the same
            # interpretation: 1 is best, more significant errors hurt more.
            score = 1.0 - 0.25 * max(sig, 0.0) - 0.05 * max(insig or 0.0, 0.0)

        if sig is not None:
            sig_errors.append(max(0.0, sig))
            type_sig_errors[qtype].append(max(0.0, sig))
            record["green_significant_errors"] = max(0.0, sig)
        if insig is not None:
            insig_errors.append(max(0.0, insig))
            type_insig_errors[qtype].append(max(0.0, insig))
            record["green_insignificant_errors"] = max(0.0, insig)
        if score is not None:
            score = max(0.0, min(1.0, score))
            green_scores.append(score)
            type_green_scores[qtype].append(score)
            record["green_score"] = score
        record["green_score_raw"] = answer
        if (idx + 1) % 25 == 0:
            log.info(f"GREEN-style judge: {idx + 1}/{sample_count} samples")

    metrics = {
        "generation_green_count": len(green_scores),
        "generation_green_score_mean": sum(green_scores) / len(green_scores) if green_scores else 0.0,
        "generation_green_significant_errors_mean": sum(sig_errors) / len(sig_errors) if sig_errors else 0.0,
        "generation_green_insignificant_errors_mean": sum(insig_errors) / len(insig_errors) if insig_errors else 0.0,
    }
    for qtype, values in type_green_scores.items():
        if values:
            metrics[f"generation_{qtype}_green_count"] = len(values)
            metrics[f"generation_{qtype}_green_score_mean"] = sum(values) / len(values)
    for qtype, values in type_sig_errors.items():
        if values:
            metrics[f"generation_{qtype}_green_significant_errors_mean"] = sum(values) / len(values)
    for qtype, values in type_insig_errors.items():
        if values:
            metrics[f"generation_{qtype}_green_insignificant_errors_mean"] = sum(values) / len(values)
    return metrics


def score_with_official_green(
    references: List[str],
    predictions: List[str],
    records: List[dict],
    model_name: str,
    output_dir: str,
    max_samples: int,
) -> Dict[str, float]:
    if not model_name or max_samples == 0:
        return {}
    sample_count = len(records) if max_samples < 0 else min(max_samples, len(records))
    if sample_count == 0:
        return {}

    try:
        from green_score import GREEN
    except Exception as exc:
        log.warning(
            "Official GREEN requested but green_score is unavailable: %s", exc)
        return {"generation_official_green_count": 0}

    refs = references[:sample_count]
    hyps = predictions[:sample_count]
    green_dir = os.path.join(output_dir, "official_green")
    os.makedirs(green_dir, exist_ok=True)

    try:
        scorer = GREEN(model_name, output_dir=green_dir)
        mean, std, green_score_list, summary, result_df = scorer(refs, hyps)
    except Exception as exc:
        log.warning("Official GREEN scoring failed: %s", exc)
        return {"generation_official_green_count": 0}

    for record, score in zip(records[:sample_count], green_score_list):
        record["official_green_score"] = float(score)

    result_path = os.path.join(green_dir, "official_green_results.csv")
    try:
        result_df.to_csv(result_path, index=False)
    except Exception as exc:
        log.warning("Could not save official GREEN result dataframe: %s", exc)

    metrics = {
        "generation_official_green_count": len(green_score_list),
        "generation_official_green_mean": float(mean),
        "generation_official_green_std": float(std),
    }
    type_scores = defaultdict(list)
    for record, score in zip(records[:sample_count], green_score_list):
        type_scores[record.get("question_type", "long_answer")].append(float(score))
    for qtype, values in type_scores.items():
        if values:
            metrics[f"generation_{qtype}_official_green_count"] = len(values)
            metrics[f"generation_{qtype}_official_green_mean"] = sum(values) / len(values)
    return metrics


def update_binary_counts(counts: Dict[str, int], y_true: int, y_pred: int):
    if y_true == 1 and y_pred == 1:
        counts["tp"] += 1
    elif y_true == 0 and y_pred == 1:
        counts["fp"] += 1
    elif y_true == 1 and y_pred == 0:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


def evaluate_generation(
    model: HyperCTVLM,
    dataset: VQADataset,
    tokenizer,
    output_dir: str,
    max_samples: int,
    max_new_tokens: int,
    num_beams: int,
    llm_score_samples: int,
    green_score_samples: int,
    judge_max_new_tokens: int,
    official_green_model: str,
    official_green_samples: int,
    stratified_eval: bool,
    samples_per_eval_bucket: int,
) -> Dict[str, float]:
    """Generate answers on held-out VQA and compute answer/clinical metrics."""
    if max_samples == 0:
        return {}

    eval_indices = select_generation_eval_indices(
        dataset,
        max_samples=max_samples,
        stratified=stratified_eval,
        samples_per_bucket=samples_per_eval_bucket,
    )
    sample_count = len(eval_indices)
    if sample_count == 0:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "vlm_generation_eval_predictions.jsonl")
    metrics_path = os.path.join(output_dir, "vlm_generation_eval_metrics.json")

    device = next(model.parameters()).device
    model.eval()
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_token_id is None or eot_token_id == tokenizer.unk_token_id:
        eot_token_id = tokenizer.eos_token_id

    exact = 0
    f1_sum = 0.0
    yes_no_question_total = 0
    yes_no_total = 0
    yes_no_correct = 0
    yes_no_parseable = 0
    yes_no_counts = defaultdict(int)
    mcq_question_total = 0
    mcq_ref_parseable = 0
    mcq_pred_parseable = 0
    mcq_correct = 0
    clinical_total = 0
    clinical_correct = 0
    clinical_counts = defaultdict(int)
    clinical_task_counts = defaultdict(lambda: defaultdict(int))
    references = []
    predictions = []
    records = []

    eval_bucket_counts = Counter()
    for eval_pos, idx in enumerate(eval_indices):
        sample = dataset.build_generation_sample(idx)
        eval_bucket = generation_eval_bucket(dataset.data[idx])
        eval_bucket_counts[eval_bucket] += 1
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        vision_tokens = [sample["vision_tokens"].to(device)]
        image_positions = [sample["image_positions"].to(device)]
        question_type = sample.get("question_type", "long_answer")

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_tokens=vision_tokens,
            image_positions=image_positions,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            eos_token_id=eot_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        raw_prediction = tokenizer.decode(
            generated[0], skip_special_tokens=False)
        prediction = clean_generated_answer(raw_prediction)
        reference = sample["reference"]
        question = sample["question"]
        is_yes_no_sample = bool(sample.get("is_yes_no_question", False))
        references.append(reference)
        predictions.append(prediction)

        norm_pred = normalize_answer(prediction)
        norm_ref = normalize_answer(reference)
        exact += int(norm_pred == norm_ref)
        f1_sum += token_f1(prediction, reference)

        ref_yn = classify_yes_no(reference)
        pred_yn = classify_yes_no(prediction)
        if is_yes_no_sample:
            yes_no_question_total += 1
            if ref_yn is not None:
                yes_no_total += 1
                if pred_yn is not None:
                    yes_no_parseable += 1
                yes_no_correct += int(pred_yn == ref_yn)
                update_binary_counts(
                    yes_no_counts,
                    ref_yn,
                    pred_yn if pred_yn is not None else 0,
                )

        ref_mcq = None
        pred_mcq = None
        if question_type == "multiple_choice":
            mcq_question_total += 1
            ref_mcq = extract_mcq_option(reference, question)
            pred_mcq = extract_mcq_option(prediction, question)
            if ref_mcq is not None:
                mcq_ref_parseable += 1
                if pred_mcq is not None:
                    mcq_pred_parseable += 1
                mcq_correct += int(pred_mcq == ref_mcq)

        question_task = infer_task_from_text(question)
        true_labels = extract_clinical_labels(reference)
        pred_labels = extract_clinical_labels(prediction)
        if question_task is not None and ref_yn is not None:
            true_labels.setdefault(question_task, ref_yn)
            if pred_yn is not None:
                pred_labels.setdefault(question_task, pred_yn)

        for task, y_true in true_labels.items():
            y_pred = pred_labels.get(task, 0)
            clinical_total += 1
            clinical_correct += int(y_true == y_pred)
            update_binary_counts(clinical_counts, y_true, y_pred)
            update_binary_counts(clinical_task_counts[task], y_true, y_pred)

        records.append({
            "id": sample["id"],
            "image": sample["image"],
            "eval_index": idx,
            "eval_bucket": eval_bucket,
            "question": question,
            "question_type": question_type,
            "question_type_label": sample.get("question_type_label", question_type),
            "is_yes_no_question": is_yes_no_sample,
            "reference": reference,
            "prediction": prediction,
            "yes_no_reference_label": ref_yn,
            "yes_no_prediction_label": pred_yn,
            "mcq_reference_option": ref_mcq,
            "mcq_prediction_option": pred_mcq,
        })

        if (eval_pos + 1) % 50 == 0:
            log.info(f"Generation eval: {eval_pos + 1}/{sample_count} samples")

    yes_no_scores = binary_scores(
        yes_no_counts["tp"], yes_no_counts["fp"],
        yes_no_counts["fn"], yes_no_counts["tn"],
    )
    clinical_scores = binary_scores(
        clinical_counts["tp"], clinical_counts["fp"],
        clinical_counts["fn"], clinical_counts["tn"],
    )
    task_f1s = []
    for task_counts in clinical_task_counts.values():
        task_scores = binary_scores(
            task_counts["tp"], task_counts["fp"],
            task_counts["fn"], task_counts["tn"],
        )
        task_f1s.append(task_scores["f1"])

    bleu1_values = [bleu1_score(pred, ref)
                    for pred, ref in zip(predictions, references)]
    rouge_l_values = [rouge_l_score(pred, ref)
                      for pred, ref in zip(predictions, references)]
    meteor_values = [meteor_score(pred, ref)
                     for pred, ref in zip(predictions, references)]
    cider_values = cider_scores(predictions, references)

    metrics = {
        "generation_samples": sample_count,
        "generation_eval_stratified": bool(stratified_eval),
        "generation_eval_bucket_counts": dict(sorted(eval_bucket_counts.items())),
        "generation_exact_match": exact / sample_count,
        "generation_token_f1": f1_sum / sample_count,
        "generation_bleu1": sum(bleu1_values) / len(bleu1_values) if bleu1_values else 0.0,
        "generation_meteor": sum(meteor_values) / len(meteor_values) if meteor_values else 0.0,
        "generation_rouge_l": sum(rouge_l_values) / len(rouge_l_values) if rouge_l_values else 0.0,
        "generation_cider": sum(cider_values) / len(cider_values) if cider_values else 0.0,
        "generation_yes_no_question_count": yes_no_question_total,
        "generation_yes_no_count": yes_no_total,
        "generation_yes_no_reference_parse_rate": yes_no_total / yes_no_question_total if yes_no_question_total else 0.0,
        "generation_yes_no_accuracy": yes_no_correct / yes_no_total if yes_no_total else 0.0,
        "generation_yes_no_parse_rate": yes_no_parseable / yes_no_total if yes_no_total else 0.0,
        "generation_yes_no_f1": yes_no_scores["f1"],
        "generation_mcq_question_count": mcq_question_total,
        "generation_mcq_classification_count": mcq_ref_parseable,
        "generation_mcq_reference_parse_rate": mcq_ref_parseable / mcq_question_total if mcq_question_total else 0.0,
        "generation_mcq_prediction_parse_rate": mcq_pred_parseable / mcq_ref_parseable if mcq_ref_parseable else 0.0,
        "generation_mcq_classification_accuracy": mcq_correct / mcq_ref_parseable if mcq_ref_parseable else 0.0,
        "generation_clinical_label_count": clinical_total,
        "generation_clinical_label_accuracy": clinical_correct / clinical_total if clinical_total else 0.0,
        "generation_clinical_micro_f1": clinical_scores["f1"],
        "generation_clinical_macro_f1": sum(task_f1s) / len(task_f1s) if task_f1s else 0.0,
        "generation_clinical_tasks_evaluated": len(task_f1s),
    }
    add_type_metrics(metrics, records, predictions, references)

    if llm_score_samples != 0:
        metrics.update(score_with_llama_judge(
            model.llm, tokenizer, records, llm_score_samples, judge_max_new_tokens))
    if green_score_samples != 0:
        metrics.update(score_with_green_style_judge(
            model.llm, tokenizer, records, green_score_samples, judge_max_new_tokens))
    if official_green_model and official_green_samples != 0:
        metrics.update(score_with_official_green(
            references, predictions, records, official_green_model,
            output_dir, official_green_samples))

    with open(predictions_path, "w") as pred_file:
        for record in records:
            pred_file.write(json.dumps(record) + "\n")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    log.info(f"Saved generation predictions to {predictions_path}")
    log.info(f"Saved generation metrics to {metrics_path}")
    for key, value in metrics.items():
        log.info(f"Generation metric | {key}: {value}")
    return metrics


def make_training_args(**kwargs) -> TrainingArguments:
    """Keep compatibility with old/new Transformers argument names."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    kwargs = {key: value for key, value in kwargs.items() if key in params}
    return TrainingArguments(**kwargs)


def configure_torchrun_device():
    """Pin each torchrun worker to its LOCAL_RANK GPU before DDP setup."""
    if not torch.cuda.is_available():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        return

    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) "
            f"are visible. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. "
            "Launch with --nproc_per_node equal to the visible GPU count."
        )

    torch.cuda.set_device(local_rank)
    log.info(
        f"Distributed worker pinned to cuda:{local_rank} "
        f"(visible_gpus={device_count}, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
    )


def main():
    parser = argparse.ArgumentParser(description="Train HyperCT VLM")
    parser.add_argument("--tokens_dir", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--val_data_json", type=str, default=None,
                        help="Optional validation VQA JSON for held-out loss/generation metrics")
    parser.add_argument("--val_tokens_dir", type=str, default=None,
                        help="Optional validation token directory; defaults to --tokens_dir")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vlm")
    parser.add_argument("--llm_name", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--llm_hidden_size", type=int, default=4096)
    parser.add_argument("--vision_dim", type=int, default=768)
    parser.add_argument("--num_queries", type=int, default=64)
    parser.add_argument("--qformer_layers", type=int, default=6)
    parser.add_argument("--qformer_heads", type=int, default=12)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--llm_lora_checkpoint", type=str, default=None,
                        help="Optional saved PEFT LoRA adapter directory for eval/resume-style loading")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_task_tokens", type=int, default=3,
                        help="Number of top tasks to select based on classifier confidence")
    parser.add_argument("--type_aware_prompts", action="store_true",
                        help="Add question-type instructions for long/short/MCQ/report samples")
    parser.add_argument("--task_hint_top_k", type=int, default=0,
                        help="Add top-k Stage 1 classifier finding hints to VQA prompts")
    parser.add_argument("--balance_question_types", action="store_true",
                        help="Oversample minority question types in the training dataset")
    parser.add_argument("--long_answer_oversample_factor", type=int, default=1,
                        help="Duplicate long-answer training samples by this factor")
    parser.add_argument("--report_generation_oversample_factor", type=int, default=1,
                        help="Duplicate report-generation training samples by this factor")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--qformer_checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and run validation/generation from saved adapters")
    parser.add_argument("--eval_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--generation_eval_samples", type=int, default=256,
                        help="Validation samples for generated-answer metrics; 0 disables, -1 uses all")
    parser.add_argument("--generation_eval_stratified", action="store_true",
                        help="Sample generation eval across long, short, yes/no, MCQ, and report buckets")
    parser.add_argument("--generation_eval_samples_per_bucket", type=int, default=0,
                        help="Samples per generation eval bucket when stratified; 0 auto-splits --generation_eval_samples")
    parser.add_argument("--generation_max_new_tokens", type=int, default=128)
    parser.add_argument("--generation_num_beams", type=int, default=1)
    parser.add_argument("--llm_score_samples", type=int, default=64,
                        help="Samples scored by the Llama-as-judge evaluator; 0 disables")
    parser.add_argument("--green_score_samples", type=int, default=0,
                        help="Samples scored by the GREEN-style radiology error judge; 0 disables")
    parser.add_argument("--judge_max_new_tokens", type=int, default=160)
    parser.add_argument("--official_green_model", type=str, default="",
                        help="Optional official green_score model, e.g. StanfordAIMI/GREEN-radllama2-7b")
    parser.add_argument("--official_green_samples", type=int, default=0,
                        help="Samples scored by official green_score package; 0 disables")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["eager", "flash_attention_2", "sdpa"])
    args = parser.parse_args()

    configure_torchrun_device()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LLM
    log.info(f"Loading LLM: {args.llm_name}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=args.attn_implementation,
    )

    # LoRA on LLM
    if args.llm_lora_checkpoint:
        log.info(f"Loading LLM LoRA adapter: {args.llm_lora_checkpoint}")
        llm = PeftModel.from_pretrained(
            llm,
            args.llm_lora_checkpoint,
            is_trainable=not args.eval_only,
        )
    else:
        target_modules = find_linear_names(llm)
        log.info(f"LoRA target modules: {target_modules}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # Q-Former adapter
    qformer = QFormerAdapter(
        vision_dim=args.vision_dim,
        llm_dim=args.llm_hidden_size,
        num_queries=args.num_queries,
        num_layers=args.qformer_layers,
        num_heads=args.qformer_heads,
    )

    if args.qformer_checkpoint:
        state = torch.load(args.qformer_checkpoint,
                           map_location="cpu", weights_only=True)
        qformer.load_state_dict(state)
        log.info(f"Loaded Q-Former checkpoint: {args.qformer_checkpoint}")

    # Dataset
    dataset = VQADataset(args.data_json, args.tokens_dir, tokenizer, args.max_length,
                         num_task_tokens=args.num_task_tokens,
                         type_aware_prompts=args.type_aware_prompts,
                         task_hint_top_k=args.task_hint_top_k,
                         balance_question_types=args.balance_question_types,
                         long_answer_oversample_factor=args.long_answer_oversample_factor,
                         report_generation_oversample_factor=args.report_generation_oversample_factor)
    log.info(f"Dataset: {len(dataset)} samples")

    eval_dataset = None
    if args.val_data_json:
        val_tokens_dir = args.val_tokens_dir or args.tokens_dir
        eval_dataset = VQADataset(
            args.val_data_json,
            val_tokens_dir,
            tokenizer,
            args.max_length,
            num_task_tokens=args.num_task_tokens,
            type_aware_prompts=args.type_aware_prompts,
            task_hint_top_k=args.task_hint_top_k,
            balance_question_types=False,
            long_answer_oversample_factor=1,
            report_generation_oversample_factor=1,
        )
        log.info(
            f"Validation dataset: {len(eval_dataset)} samples | tokens={val_tokens_dir}")

    # Training args
    eval_strategy = args.eval_strategy if eval_dataset is not None else "no"
    eval_batch_size = args.eval_batch_size or args.batch_size
    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        report_to="none",
        gradient_checkpointing=True,
        eval_strategy=eval_strategy,
        prediction_loss_only=True,
    )
    if eval_strategy == "steps" and args.eval_steps is not None:
        training_kwargs["eval_steps"] = args.eval_steps
    training_args = make_training_args(**training_kwargs)

    # Combine Q-Former + LLM into single model
    model = HyperCTVLM(llm, qformer)

    trainer = PerplexityTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    if args.eval_only:
        log.info("eval_only=True: skipping training and running evaluation/generation only.")
    else:
        trainer.train()

    if eval_dataset is not None:
        final_metrics = trainer.evaluate(metric_key_prefix="final_eval")
        if trainer.is_world_process_zero():
            log.info(f"Final validation metrics: {final_metrics}")

    if (
        eval_dataset is not None
        and args.generation_eval_samples != 0
        and trainer.is_world_process_zero()
    ):
        eval_model = trainer.model
        if hasattr(eval_model, "module"):
            eval_model = eval_model.module
        evaluate_generation(
            model=eval_model,
            dataset=eval_dataset,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_samples=args.generation_eval_samples,
            max_new_tokens=args.generation_max_new_tokens,
            num_beams=args.generation_num_beams,
            llm_score_samples=args.llm_score_samples,
            green_score_samples=args.green_score_samples,
            judge_max_new_tokens=args.judge_max_new_tokens,
            official_green_model=args.official_green_model,
            official_green_samples=args.official_green_samples,
            stratified_eval=args.generation_eval_stratified,
            samples_per_eval_bucket=args.generation_eval_samples_per_bucket,
        )

    if trainer.is_world_process_zero() and not args.eval_only:
        # Save Q-Former separately
        qformer_path = os.path.join(args.output_dir, "qformer_final.pt")
        torch.save(qformer.state_dict(), qformer_path)
        log.info(f"Saved Q-Former to {qformer_path}")

        # Save LoRA adapter
        llm.save_pretrained(os.path.join(args.output_dir, "llm_lora"))
        log.info("Training complete.")


if __name__ == "__main__":
    main()
