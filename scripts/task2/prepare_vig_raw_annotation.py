#!/usr/bin/env python3
"""
Prepare a manual annotation sheet for Task 2 using ViG raw predictions only.

This script does not assign error labels. It only collects images whose
references contain train-only V_cultural terms and writes blank columns for
human annotation.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TASK_ROOT.parent
TASK1_FILE = TASK_ROOT / "output" / "task1" / "task1_cultural_entity_recall.json"
TEST_DATA_FILE = REPO_ROOT / "data" / "test_data.json"
PREDICTION_FILE = TASK_ROOT / "data" / "predictions" / "vig_raw_epoch10_details.json"
IMAGE_DIR = REPO_ROOT / "data" / "public-test-images"
OUTPUT_CSV = TASK_ROOT / "output" / "task2" / "task2_vig_raw_annotation_template.csv"
OUTPUT_JSONL = TASK_ROOT / "output" / "task2" / "task2_vig_raw_annotation_template.jsonl"
SAMPLE_SIZE = 100


def normalize_image_id(image_id):
    try:
        return int(image_id)
    except (TypeError, ValueError):
        return image_id


def load_test_data():
    data = json.loads(TEST_DATA_FILE.read_text(encoding="utf-8"))
    image_files = {img["id"]: img["filename"] for img in data["images"]}

    refs_by_img = defaultdict(list)
    for ann in data["annotations"]:
        refs_by_img[ann["image_id"]].append(
            ann.get("segment_caption") or ann.get("caption", "")
        )

    return image_files, refs_by_img


def load_predictions():
    rows = json.loads(PREDICTION_FILE.read_text(encoding="utf-8"))
    return {
        normalize_image_id(row["image_id"]): row.get("prediction", "")
        for row in rows
    }


def load_concept_images():
    task1 = json.loads(TASK1_FILE.read_text(encoding="utf-8"))
    concepts = task1["concepts"]

    concept_images = {}
    facets = {}
    for concept, info in concepts.items():
        images = [normalize_image_id(image_id) for image_id in info["images"]]
        if images:
            concept_images[concept] = images
            facets[concept] = info["facet"]

    return concept_images, facets


def select_images(concept_images):
    selected = []
    seen = set()
    concept_order = sorted(
        concept_images,
        key=lambda concept: (-len(concept_images[concept]), concept),
    )

    max_len = max(len(images) for images in concept_images.values())
    for index in range(max_len):
        for concept in concept_order:
            images = concept_images[concept]
            if index >= len(images):
                continue
            image_id = images[index]
            if image_id in seen:
                continue
            selected.append(image_id)
            seen.add(image_id)
            if len(selected) >= SAMPLE_SIZE:
                return selected

    return selected


def build_terms_by_image(concept_images, facets):
    terms_by_img = defaultdict(list)
    facets_by_img = defaultdict(list)

    for concept, images in concept_images.items():
        for image_id in images:
            if concept not in terms_by_img[image_id]:
                terms_by_img[image_id].append(concept)
            facet = facets[concept]
            if facet not in facets_by_img[image_id]:
                facets_by_img[image_id].append(facet)

    return terms_by_img, facets_by_img


def write_outputs(rows):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "image_id",
        "image_file",
        "image_path",
        "reference_cultural_terms",
        "reference_facets",
        "references",
        "vig_prediction",
        "cultural_entity_missed",
        "template_bias",
        "object_hallucination",
        "notes",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    image_files, refs_by_img = load_test_data()
    predictions = load_predictions()
    concept_images, facets = load_concept_images()
    selected_images = select_images(concept_images)
    terms_by_img, facets_by_img = build_terms_by_image(concept_images, facets)

    rows = []
    for sample_index, image_id in enumerate(selected_images, start=1):
        image_file = image_files.get(image_id, "")
        row = {
            "sample_id": sample_index,
            "image_id": image_id,
            "image_file": image_file,
            "image_path": str(IMAGE_DIR / image_file) if image_file else "",
            "reference_cultural_terms": "; ".join(terms_by_img[image_id]),
            "reference_facets": "; ".join(facets_by_img[image_id]),
            "references": " ||| ".join(refs_by_img[image_id]),
            "vig_prediction": predictions.get(image_id, ""),
            "cultural_entity_missed": "",
            "template_bias": "",
            "object_hallucination": "",
            "notes": "",
        }
        rows.append(row)

    write_outputs(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"Wrote {len(rows)} rows to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
