#!/usr/bin/env python3
"""
Create an AI/heuristic draft annotation for Task 2 using ViG raw predictions.

This is only a review aid. Final Task 2 labels should be checked by a human
against the actual image. For object_hallucination, the script writes REVIEW
when text-only evidence is not enough.
"""

import csv
import json
import re
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[2]
VOCAB_FILE = TASK_ROOT / "output" / "task0" / "V_cultural_train_only.json"
INPUT_CSV = TASK_ROOT / "output" / "task2" / "task2_vig_raw_annotation_template.csv"
OUTPUT_CSV = TASK_ROOT / "output" / "task2" / "task2_vig_raw_annotation_ai_draft.csv"
OUTPUT_JSONL = TASK_ROOT / "output" / "task2" / "task2_vig_raw_annotation_ai_draft.jsonl"

GENERIC_CUES = [
    "bức ảnh",
    "khung_cảnh",
    "sự xuất_hiện",
    "có sự xuất_hiện",
    "đang xuất_hiện",
    "xuất_hiện ở trong",
    "xuất_hiện trong",
    "có nhiều người",
    "có một người",
    "có một cô gái",
    "có một người phụ_nữ",
    "có một người đàn_ông",
    "đang đứng",
    "đang ngồi",
    "tạo_dáng chụp ảnh",
]

STOPWORDS = {
    "có", "một", "những", "các", "đang", "ở", "trên", "trong", "và", "của",
    "là", "đây", "phía", "trước", "sau", "bên", "với", "vào", "ra", "nhiều",
    "xuất_hiện", "sự", "bức", "ảnh", "khung_cảnh", "người", "màu", "được",
    "đang", "đứng", "ngồi", "di_chuyển", "đi", "chụp", "tạo_dáng",
}

STRONG_OBJECTS = {
    "áo_dài", "bậc thang", "bàn_ghế", "biển", "căn nhà", "cây cầu", "cửa_hàng",
    "đèn_lồng", "đường", "ghe", "ghế", "khu chợ", "khu di_tích", "mặt_nước",
    "nón lá", "quầy hàng", "sông", "thuyền", "toà nhà", "vỉa_hè", "xe_đạp",
    "xe_máy", "xích_lô", "xuồng", "cờ", "lá cờ", "nhà_thờ", "ngôi chùa",
    "bó hoa", "hoa", "học_sinh",
}


def load_vocab():
    vocab = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    return {
        term: {
            "facet": meta.get("facet", ""),
            "variants": meta.get("variants", []),
        }
        for term, meta in vocab.items()
    }


def concept_forms(term, meta):
    forms = {term, term.replace("_", " ")}
    for variant in meta.get("variants", []):
        forms.add(variant)
        forms.add(variant.replace("_", " "))
    return sorted(forms, key=len, reverse=True)


def contains_any(text, forms):
    return any(form in text for form in forms)


def split_terms(value):
    return [term.strip() for term in value.split(";") if term.strip()]


def extract_prediction_objects(prediction):
    found = []
    for obj in sorted(STRONG_OBJECTS, key=len, reverse=True):
        if obj in prediction and obj not in found:
            found.append(obj)
    return found


def tokens(text):
    return re.findall(r"[\w_]+", text, flags=re.UNICODE)


def salient_missing_objects(prediction, references):
    refs = references.replace("|||", " ")
    missing = []

    for obj in extract_prediction_objects(prediction):
        if obj not in refs and obj.replace("_", " ") not in refs:
            missing.append(obj)

    return missing[:6]


def autolabel_row(row, vocab):
    prediction = row["vig_prediction"]
    references = row["references"]
    reference_terms = split_terms(row["reference_cultural_terms"])

    recovered_terms = []
    missing_terms = []
    for term in reference_terms:
        meta = vocab.get(term, {"variants": []})
        forms = concept_forms(term, meta)
        if contains_any(prediction, forms):
            recovered_terms.append(term)
        else:
            missing_terms.append(term)

    cultural_entity_missed = "Y" if missing_terms else "N"

    generic_count = sum(1 for cue in GENERIC_CUES if cue in prediction)
    pred_cultural_terms = [
        term for term, meta in vocab.items()
        if contains_any(prediction, concept_forms(term, meta))
    ]
    short_prediction = len(tokens(prediction)) <= 10
    template_bias = "Y" if (
        cultural_entity_missed == "Y"
        and generic_count > 0
        and (short_prediction or not pred_cultural_terms)
    ) else "N"

    possible_hallucinations = salient_missing_objects(prediction, references)
    hallucination_terms = [
        term for term in possible_hallucinations
        if term not in pred_cultural_terms and term not in recovered_terms
    ]
    object_hallucination = "REVIEW" if hallucination_terms else "N"

    reasons = []
    if recovered_terms:
        reasons.append("recover=" + ",".join(recovered_terms))
    if missing_terms:
        reasons.append("miss=" + ",".join(missing_terms))
    if generic_count:
        reasons.append(f"generic_cues={generic_count}")
    if hallucination_terms:
        reasons.append("possible_hallucination=" + ",".join(hallucination_terms))
    reasons.append("AI draft; cần duyệt bằng ảnh")

    row["cultural_entity_missed"] = cultural_entity_missed
    row["template_bias"] = template_bias
    row["object_hallucination"] = object_hallucination
    row["notes"] = "; ".join(reasons)
    row["draft_recovered_terms"] = "; ".join(recovered_terms)
    row["draft_missing_terms"] = "; ".join(missing_terms)
    row["draft_possible_hallucinated_terms"] = "; ".join(hallucination_terms)
    return row


def main():
    vocab = load_vocab()
    with INPUT_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    labeled = [autolabel_row(dict(row), vocab) for row in rows]

    fieldnames = list(labeled[0].keys())
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(labeled)} draft labels to {OUTPUT_CSV}")
    print(f"Wrote {len(labeled)} draft labels to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
