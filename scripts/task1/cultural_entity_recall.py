#!/usr/bin/env python3
"""
Task 1: Cultural Entity Recall Probe
- Find images referencing cultural concepts
- Check if models mention those concepts
- Calculate recall per concept and per model
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

TASK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TASK_ROOT.parent
VOCAB_FILE = TASK_ROOT / "output" / "task0" / "V_cultural_train_only.json"
MODEL_FILES = {
    "vig_raw_epoch10": TASK_ROOT / "data" / "predictions" / "vig_raw_epoch10_details.json",
}

# Fallback seed cultural vocabulary from note.
SEED_CULTURAL_VOCAB = {
    "nón_lá": "trang_phục",
    "áo_dài": "trang_phục",
    "áo_bà_ba": "trang_phục",
    "xích_lô": "phương_tiện",
    "mặt_nạ": "trang_trí",
    "cờ": "biểu_tượng",
    "cờ_đỏ_sao_vàng": "biểu_tượng",
    "trống": "nghi_lễ",
    "trống_đồng": "nghi_lễ",
    "đèn_lồng": "trang_trí",
    "phở": "ẩm_thực",
    "bánh_mì": "ẩm_thực",
}

def load_cultural_vocab():
    """Load audited train-only V_cultural if available; otherwise use seed vocab."""
    if VOCAB_FILE.exists():
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return {
            term: {
                "facet": meta.get("facet", "unknown"),
                "variants": meta.get("variants", [])
            }
            for term, meta in vocab.items()
        }

    return {
        term: {
            "facet": facet,
            "variants": []
        }
        for term, facet in SEED_CULTURAL_VOCAB.items()
    }

def concept_forms(term, meta):
    """Return exact-match surface forms for one concept."""
    forms = {term, term.replace("_", " ")}
    for variant in meta.get("variants", []):
        forms.add(variant)
        forms.add(variant.replace("_", " "))
    return sorted(forms, key=len, reverse=True)

def contains_concept(text, forms):
    return any(
        re.search(rf"(?<![\w]){re.escape(form)}(?![\w])", text)
        for form in forms
    )

def normalize_image_id(image_id):
    try:
        return int(image_id)
    except (TypeError, ValueError):
        return image_id

def load_model_outputs():
    """Load available model outputs into {model_name: {image_id: caption}}."""
    models = {}

    for model_name, model_file in MODEL_FILES.items():
        if not model_file.exists():
            print(f"    ! Missing {model_name}: {model_file}")
            continue

        with open(model_file, 'r', encoding='utf-8') as f:
            rows = json.load(f)

        output_by_img = {}
        for item in rows:
            image_id = normalize_image_id(item.get("image_id"))
            caption = item.get("caption") or item.get("prediction") or ""
            output_by_img[image_id] = caption

        models[model_name] = output_by_img
        print(f"    ✓ Loaded {model_name}: {len(output_by_img)} predictions")

    return models

def load_data():
    """Load test data with references and baseline captions"""
    with open(REPO_ROOT / "data" / "test_data.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Build image -> references mapping
    img_refs = defaultdict(list)
    img_ids = {}
    
    for img in test_data['images']:
        img_ids[img['id']] = img['filename']
    
    for ann in test_data['annotations']:
        caption = ann.get('segment_caption') or ann.get('caption', '')
        img_id = ann['image_id']
        if img_id not in [img['id'] for img in test_data['images']]:
            continue
        img_refs[img_id].append(caption)
    
    return img_refs, img_ids, test_data

def find_images_with_concept(img_refs, cultural_vocab):
    """
    Find images whose references mention given concepts.
    For now, use simple string matching.
    """
    images_with_concept = defaultdict(list)
    
    for concept, meta in cultural_vocab.items():
        forms = concept_forms(concept, meta)
        for img_id, refs in img_refs.items():
            # Check if concept appears in any reference
            for ref in refs:
                if contains_concept(ref, forms):
                    images_with_concept[concept].append(img_id)
                    break
    
    return images_with_concept

def calculate_recall(img_refs, cultural_vocab, model_outputs):
    """
    Calculate reference coverage and per-model cultural recall.
    """
    results = {}
    
    for concept, meta in cultural_vocab.items():
        forms = concept_forms(concept, meta)
        matching_imgs = []
        for img_id, refs in img_refs.items():
            for ref in refs:
                if contains_concept(ref, forms):
                    matching_imgs.append(img_id)
                    break

        model_results = {}
        matching_set = set(matching_imgs)
        for model_name, outputs in model_outputs.items():
            matched_imgs = []
            missed_imgs = []
            for img_id in matching_imgs:
                caption = outputs.get(img_id, "")
                if contains_concept(caption, forms):
                    matched_imgs.append(img_id)
                else:
                    missed_imgs.append(img_id)

            total = len(matching_set)
            model_results[model_name] = {
                "matched": len(matched_imgs),
                "total": total,
                "recall": (len(matched_imgs) / total) if total else 0.0,
                "matched_images": matched_imgs,
                "missed_images": missed_imgs
            }
        
        results[concept] = {
            "facet": meta.get("facet", "unknown"),
            "forms": forms,
            "total_occurrences": len(matching_imgs),
            "images": matching_imgs,
            "models": model_results
        }
    
    return results

def main():
    print("\n=== Task 1: Cultural Entity Recall Probe ===\n")
    
    # Load data
    print("[1] Loading test data...")
    img_refs, img_ids, test_data = load_data()
    print(f"    ✓ Loaded {len(img_refs)} test images")
    print(f"    ✓ References loaded")

    print("\n[2] Loading model outputs...")
    model_outputs = load_model_outputs()
    
    # Find images with cultural concepts
    print("\n[3] Finding images with cultural concepts...")
    cultural_vocab = load_cultural_vocab()
    concepts = list(cultural_vocab.keys())
    print(f"    ✓ Loaded {len(concepts)} concepts from {VOCAB_FILE if VOCAB_FILE.exists() else 'seed vocabulary'}")
    images_with_concept = find_images_with_concept(img_refs, cultural_vocab)
    
    print(f"    ✓ Found {len(images_with_concept)} concepts in references")
    for concept, imgs in images_with_concept.items():
        facet = cultural_vocab[concept]["facet"]
        print(f"      - {concept:<20} ({facet:<15}): {len(imgs)} images")
    
    # Calculate recall
    print("\n[4] Calculating recall...")
    recall_results = calculate_recall(img_refs, cultural_vocab, model_outputs)
    
    # Save results
    output_dir = TASK_ROOT / "output" / "task1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "task1_cultural_entity_recall.json"
    metadata = {
        "vocab_file": str(VOCAB_FILE.relative_to(REPO_ROOT)),
        "prediction_files": {
            model_name: str(model_file.relative_to(REPO_ROOT))
            for model_name, model_file in MODEL_FILES.items()
        },
        "models": list(model_outputs.keys()),
        "concepts": recall_results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"    ✓ Saved to {output_file}")
    
    # Summary
    print(f"\n[5] Summary:")
    print(f"    - Total cultural concepts: {len(concepts)}")
    print(f"    - Concepts found in references: {len(images_with_concept)}")
    print(f"    - Images with cultural references: {sum(len(imgs) for imgs in images_with_concept.values())}")
    for model_name in model_outputs:
        matched = 0
        total = 0
        for result in recall_results.values():
            model_result = result["models"][model_name]
            matched += model_result["matched"]
            total += model_result["total"]
        recall = matched / total if total else 0.0
        print(f"    - {model_name}: {matched}/{total} concept-image matches ({recall:.4f})")
    
    print(f"\nNote: This result uses train-only V_cultural when {VOCAB_FILE.name} exists.")
    print(f"\nNext: Task 2 - Error Taxonomy (requires manual annotation)")

if __name__ == '__main__':
    main()
