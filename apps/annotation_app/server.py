#!/usr/bin/env python3
"""Local ViG raw annotation app.

No external dependencies. Serves the UI, test images, prediction details, and
persists manual annotations to task/output/task2.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
TASK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TASK_ROOT.parent
STATIC_DIR = APP_DIR / "static"
PREDICTION_FILE = TASK_ROOT / "data" / "predictions" / "vig_raw_epoch10_details.json"
TEST_DATA_FILE = REPO_ROOT / "data" / "test_data.json"
VOCAB_FILE = TASK_ROOT / "output" / "task0" / "V_cultural_train_only.json"
IMAGE_DIR = REPO_ROOT / "data" / "public-test-images"
OUTPUT_JSON = TASK_ROOT / "output" / "task2" / "vig_raw_annotations.json"
OUTPUT_CSV = TASK_ROOT / "output" / "task2" / "vig_raw_annotations.csv"

VIETNAMESE_STOPWORDS = {
    "có", "một", "những", "các", "ở", "trong", "trên", "dưới", "phía", "trước",
    "sau", "bên", "và", "là", "của", "với", "đang", "được", "vào", "ra", "này",
    "kia", "ảnh", "bức", "hình", "khung_cảnh", "xuất_hiện", "sự", "nhiều", "vài",
    "màu", "người", "phụ_nữ", "đàn_ông", "cô", "gái", "chàng", "trai", "đây",
    "đó", "chiếc", "cái", "con", "đang", "được", "ra", "về", "từ", "cho", "vào",
    "lên", "xuống", "đi", "nằm", "đứng", "ngồi", "bên", "giữa", "xung_quanh",
}

TEMPLATE_PATTERNS = [
    "sự xuất_hiện",
    "xuất_hiện ở trong bức ảnh",
    "ở trong bức ảnh",
    "có sự xuất_hiện",
    "có những",
    "có một",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_image_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image_map() -> dict[int, str]:
    test_data = load_json(TEST_DATA_FILE, {"images": []})
    return {normalize_image_id(item["id"]): item["filename"] for item in test_data.get("images", [])}


def concept_forms(term: str, meta: dict) -> list[str]:
    forms = {term, term.replace("_", " ")}
    for variant in meta.get("variants", []):
        forms.add(variant)
        forms.add(variant.replace("_", " "))
    return sorted(forms, key=len, reverse=True)


def detect_terms(texts: list[str], vocab: dict) -> list[dict]:
    joined = "\n".join(texts)
    detected = []
    for term, meta in vocab.items():
        forms = concept_forms(term, meta)
        matched_form = next((form for form in forms if form and phrase_in_text(joined, form)), None)
        if matched_form:
            detected.append({
                "term": term,
                "facet": meta.get("facet", ""),
                "matched_form": matched_form,
            })
    return detected


def phrase_in_text(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return bool(re.search(rf"(?<![\w]){re.escape(phrase)}(?![\w])", text))


def term_names(terms: list[dict]) -> list[str]:
    return [item["term"] for item in terms]


def token_list(text: str) -> list[str]:
    cleaned = text.lower()
    for char in ",.;:!?()[]{}\"'":
        cleaned = cleaned.replace(char, " ")
    return [token for token in cleaned.split() if token and token not in VIETNAMESE_STOPWORDS]


def tokenize(text: str) -> set[str]:
    return set(token_list(text))


def top_keywords(texts: list[str], limit: int = 8) -> list[str]:
    counts = Counter()
    for text in texts:
        counts.update(token_list(text))
    return [token for token, _ in counts.most_common(limit)]


def join_terms(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " và " + values[-1]


def max_reference_overlap(prediction: str, references: list[str]) -> float:
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    overlaps = []
    for ref in references:
        ref_tokens = tokenize(ref)
        if not ref_tokens:
            continue
        overlaps.append(len(pred_tokens & ref_tokens) / max(1, len(pred_tokens | ref_tokens)))
    return max(overlaps, default=0.0)


def is_template_like(prediction: str) -> bool:
    hits = sum(1 for pattern in TEMPLATE_PATTERNS if pattern in prediction)
    token_len = len(prediction.split())
    return hits >= 2 or (hits >= 1 and token_len <= 11)


def build_comparison_comment(
    item: dict,
    expected_terms: list[str],
    predicted_terms: list[str],
    missing_terms: list[str],
    extra_terms: list[str],
    overlap: float,
    template_bias: bool,
) -> str:
    prediction = item["prediction"]
    references = item["references"]
    ref_keywords = top_keywords(references, limit=10)
    pred_keywords = top_keywords([prediction], limit=8)
    missing_keywords = [token for token in ref_keywords if token not in pred_keywords][:5]
    extra_keywords = [token for token in pred_keywords if token not in ref_keywords][:5]
    shared_terms = [term for term in expected_terms if term in predicted_terms]
    shared_keywords = [token for token in ref_keywords if token in pred_keywords][:4]

    clauses = []
    if shared_terms:
        clauses.append("ViG mô tả đúng cultural term " + join_terms(shared_terms))
    elif shared_keywords:
        clauses.append("ViG mô tả đúng một số chi tiết như " + join_terms(shared_keywords))
    else:
        clauses.append("ViG chưa khớp rõ với các chi tiết chính trong ảnh")

    if missing_terms:
        clauses.append("bỏ sót cultural term " + join_terms(missing_terms))
    if missing_keywords:
        clauses.append("bỏ sót/giảm nhẹ các chi tiết " + join_terms(missing_keywords))
    if extra_terms:
        clauses.append("sinh thêm cultural term " + join_terms(extra_terms) + " không phù hợp với ảnh")
    if extra_keywords:
        clauses.append("lệch sang các chi tiết " + join_terms(extra_keywords))

    note = "; ".join(clauses).strip()
    if not note.endswith("."):
        note += "."
    return note


def make_draft_annotation(item: dict) -> dict:
    expected_terms = term_names(item["reference_terms"])
    predicted_terms = term_names(item["prediction_terms"])
    missing_terms = sorted(set(expected_terms) - set(predicted_terms))
    extra_terms = sorted(set(predicted_terms) - set(expected_terms))
    overlap = max_reference_overlap(item["prediction"], item["references"])
    template_like = is_template_like(item["prediction"])
    cultural_missed = bool(missing_terms)
    template_bias = template_like and (cultural_missed or overlap < 0.22)

    if overlap >= 0.42 and not cultural_missed:
        caption_quality = "correct"
    elif overlap >= 0.20 or predicted_terms or expected_terms:
        caption_quality = "partial"
    else:
        caption_quality = "unsure"

    specificity = "generic" if template_bias else "specific"
    needs_review = cultural_missed or extra_terms or overlap < 0.20

    explanation = build_comparison_comment(
        item,
        expected_terms,
        predicted_terms,
        missing_terms,
        extra_terms,
        overlap,
        template_bias,
    )

    return {
        "source": "heuristic_draft_review_required",
        "labels": {
            "reviewed": False,
            "needs_review": needs_review,
            "caption_quality": caption_quality,
            "specificity": specificity,
            "cultural_entity_missed": cultural_missed,
            "template_bias": template_bias,
            "object_hallucination": False,
            "wrong_object_or_action": False,
            "language_issue": bool(item.get("contains_unk") or item.get("duplicate_ngram_flag")),
        },
        "expected_cultural_terms": expected_terms,
        "predicted_cultural_terms": predicted_terms,
        "missing_cultural_terms": missing_terms,
        "extra_cultural_terms": extra_terms,
        "reference_overlap": round(overlap, 4),
        "explanation": explanation,
        "caveat": "Draft này không phải ground truth; annotator cần nhìn ảnh và duyệt trước khi Save.",
    }


def load_items() -> list[dict]:
    predictions = load_json(PREDICTION_FILE, [])
    image_map = load_image_map()
    vocab = load_json(VOCAB_FILE, {})

    items = []
    for row in predictions:
        image_id = normalize_image_id(row.get("image_id"))
        filename = image_map.get(image_id, f"{image_id:011d}.jpg" if isinstance(image_id, int) else "")
        prediction = row.get("prediction", "")
        references = row.get("references", [])
        ref_terms = detect_terms(references, vocab)
        pred_terms = detect_terms([prediction], vocab)
        item = {
            "image_id": image_id,
            "image_filename": filename,
            "image_url": f"/images/{filename}",
            "prediction": prediction,
            "references": references,
            "reference_terms": ref_terms,
            "prediction_terms": pred_terms,
            "prediction_token_len": row.get("prediction_token_len"),
            "reference_token_len_mean": row.get("reference_token_len_mean"),
            "contains_unk": row.get("contains_unk"),
            "duplicate_ngram_flag": row.get("duplicate_ngram_flag"),
        }
        item["ai_draft"] = make_draft_annotation(item)
        items.append(item)

    return sorted(items, key=lambda item: item["image_id"])


def empty_store() -> dict:
    return {
        "metadata": {
            "schema_version": 1,
            "task": "vig_raw_manual_annotation",
            "source_predictions": str(PREDICTION_FILE.relative_to(REPO_ROOT)),
            "vocab": str(VOCAB_FILE.relative_to(REPO_ROOT)),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        },
        "annotations": [],
    }


def load_store() -> dict:
    store = load_json(OUTPUT_JSON, None)
    if not store:
        return empty_store()
    store.setdefault("metadata", empty_store()["metadata"])
    store.setdefault("annotations", [])
    return store


def annotation_index(store: dict) -> dict[str, dict]:
    return {str(item.get("image_id")): item for item in store.get("annotations", [])}


def clean_removed_labels(labels: dict) -> dict:
    cleaned = dict(labels or {})
    cleaned.pop("missing_objects", None)
    return cleaned


def write_outputs(store: dict) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    store["metadata"]["updated_at"] = utc_now()

    tmp = OUTPUT_JSON.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, OUTPUT_JSON)

    csv_fields = [
        "image_id",
        "image_filename",
        "reviewed",
        "needs_review",
        "caption_quality",
        "specificity",
        "cultural_entity_missed",
        "template_bias",
        "object_hallucination",
        "wrong_object_or_action",
        "language_issue",
        "expected_cultural_terms",
        "predicted_cultural_terms",
        "explanation",
        "updated_at",
    ]
    tmp_csv = OUTPUT_CSV.with_suffix(".csv.tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for ann in sorted(store.get("annotations", []), key=lambda item: normalize_image_id(item.get("image_id"))):
            labels = clean_removed_labels(ann.get("labels", {}))
            writer.writerow({
                "image_id": ann.get("image_id", ""),
                "image_filename": ann.get("image_filename", ""),
                "reviewed": labels.get("reviewed", False),
                "needs_review": labels.get("needs_review", False),
                "caption_quality": labels.get("caption_quality", ""),
                "specificity": labels.get("specificity", ""),
                "cultural_entity_missed": labels.get("cultural_entity_missed", False),
                "template_bias": labels.get("template_bias", False),
                "object_hallucination": labels.get("object_hallucination", False),
                "wrong_object_or_action": labels.get("wrong_object_or_action", False),
                "language_issue": labels.get("language_issue", False),
                "expected_cultural_terms": ";".join(ann.get("expected_cultural_terms", [])),
                "predicted_cultural_terms": ";".join(ann.get("predicted_cultural_terms", [])),
                "explanation": ann.get("explanation", ""),
                "updated_at": ann.get("updated_at", ""),
            })
    os.replace(tmp_csv, OUTPUT_CSV)


def save_annotation(payload: dict) -> dict:
    store = load_store()
    indexed = annotation_index(store)
    image_id = normalize_image_id(payload.get("image_id"))
    key = str(image_id)
    now = utc_now()
    labels = clean_removed_labels(payload.get("labels", {}))
    ai_draft = dict(payload.get("ai_draft", {}) or {})
    ai_draft["labels"] = clean_removed_labels(ai_draft.get("labels", {}))

    annotation = {
        "image_id": image_id,
        "image_filename": payload.get("image_filename", ""),
        "prediction": payload.get("prediction", ""),
        "references": payload.get("references", []),
        "reference_terms": payload.get("reference_terms", []),
        "prediction_terms": payload.get("prediction_terms", []),
        "ai_draft": ai_draft,
        "labels": labels,
        "expected_cultural_terms": payload.get("expected_cultural_terms", []),
        "predicted_cultural_terms": payload.get("predicted_cultural_terms", []),
        "explanation": payload.get("explanation", ""),
        "annotator": payload.get("annotator", ""),
        "updated_at": now,
    }

    if key in indexed:
        indexed[key].update(annotation)
    else:
        store["annotations"].append(annotation)

    write_outputs(store)
    return {
        "ok": True,
        "saved_at": now,
        "progress": progress_summary(store),
        "paths": {
            "json": str(OUTPUT_JSON.relative_to(REPO_ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(REPO_ROOT)),
        },
    }


def progress_summary(store: dict) -> dict:
    annotations = store.get("annotations", [])
    reviewed = sum(1 for ann in annotations if ann.get("labels", {}).get("reviewed"))
    needs_review = sum(1 for ann in annotations if ann.get("labels", {}).get("needs_review"))
    return {
        "saved": len(annotations),
        "reviewed": reviewed,
        "needs_review": needs_review,
    }


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "ViGAnnotationApp/1.0"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.send_file(STATIC_DIR / "index.html")
            return

        if path == "/api/items":
            items = load_items()
            store = load_store()
            annotations = annotation_index(store)
            for item in items:
                item["annotation"] = annotations.get(str(item["image_id"]))
            self.send_json({
                "items": items,
                "progress": progress_summary(store),
                "outputs": {
                    "json": str(OUTPUT_JSON.relative_to(REPO_ROOT)),
                    "csv": str(OUTPUT_CSV.relative_to(REPO_ROOT)),
                },
            })
            return

        if path == "/api/progress":
            self.send_json(progress_summary(load_store()))
            return

        if path == "/api/export.json":
            self.send_file(OUTPUT_JSON)
            return

        if path == "/api/export.csv":
            self.send_file(OUTPUT_CSV)
            return

        if path.startswith("/images/"):
            filename = Path(unquote(path.removeprefix("/images/"))).name
            self.send_file(IMAGE_DIR / filename)
            return

        if path.startswith("/static/"):
            filename = Path(unquote(path.removeprefix("/static/"))).name
            self.send_file(STATIC_DIR / filename)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotation":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = save_annotation(payload)
            self.send_json(result)
        except Exception as exc:  # noqa: BLE001 - return useful local error
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"ViG annotation app: http://{args.host}:{args.port}")
    print(f"Saving JSON: {OUTPUT_JSON}")
    print(f"Saving CSV:  {OUTPUT_CSV}")
    server.serve_forever()


if __name__ == "__main__":
    main()
