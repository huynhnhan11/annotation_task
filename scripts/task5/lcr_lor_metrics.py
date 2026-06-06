#!/usr/bin/env python3
"""
Task 5: LCR/LOR Diagnostic Metrics
- Tính Local Cultural Recall (LCR)
- Tính Local Overall Recall (LOR)
- Dùng exact/variant matching với V_cultural train-only
- Đo output ViG thô

Requires V_cultural từ Task 0 Step 3-4
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

# Fallback seed cultural vocabulary.
SEED_CULTURAL_VOCAB = {
    "nón_lá": "trang_phục",
    "áo_dài": "trang_phục",
    "áo_bà_ba": "trang_phục",
    "xích_lô": "phương_tiện",
    "mặt_nạ": "trang_trí",
    "cờ": "biểu_tượng",
    "cờ_đỏ_sao_vàng": "biểu_tượng",
    "trống": "nghi_lễ",
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
    """Load model outputs (baseline and ViG captions)"""
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
        print(f"    ✓ Loaded {model_name}: {len(output_by_img)} items")
    
    return models

def load_test_references():
    """Load test references"""
    with open(REPO_ROOT / "data" / "test_data.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    refs_by_img = defaultdict(list)
    for ann in test_data['annotations']:
        caption = ann.get('segment_caption') or ann.get('caption', '')
        refs_by_img[ann['image_id']].append(caption)
    
    return refs_by_img

def calculate_lcr_lor(output_by_img, refs, cultural_vocab):
    """
    Calculate LCR and LOR for a model.
    - LCR: Cultural Recall = % of cultural terms in references that appear in captions
    - LOR: Overall Recall = generic recall metric
    """
    results = {
        'total_images': 0,
        'cultural_matches': 0,
        'total_cultural_refs': 0,
        'lcr': 0.0,
        'details_per_image': []
    }
    
    for img_id, ref_captions in refs.items():
        if img_id not in output_by_img:
            continue
        
        results['total_images'] += 1
        model_caption = output_by_img[img_id]
        
        # Check cultural terms
        image_details = {
            'image_id': img_id,
            'cultural_matches': []
        }
        
        for cultural_term, meta in cultural_vocab.items():
            forms = concept_forms(cultural_term, meta)
            # Check if cultural term appears in any reference
            found_in_ref = any(contains_concept(ref, forms) for ref in ref_captions)
            
            # Check if cultural term appears in model output
            found_in_output = contains_concept(model_caption, forms)
            
            if found_in_ref:
                results['total_cultural_refs'] += 1
                if found_in_output:
                    results['cultural_matches'] += 1
                    image_details['cultural_matches'].append(cultural_term)
        
        results['details_per_image'].append(image_details)
    
    if results['total_cultural_refs'] > 0:
        results['lcr'] = results['cultural_matches'] / results['total_cultural_refs']
    
    return results

def main():
    print("\n=== Task 5: LCR/LOR Diagnostic Metrics ===\n")
    
    print("[1] Loading data...")
    model_outputs = load_model_outputs()
    refs = load_test_references()
    cultural_vocab = load_cultural_vocab()
    print(f"    ✓ Loaded {len(refs)} test references")
    print(f"    ✓ Loaded {len(cultural_vocab)} cultural concepts from {VOCAB_FILE if VOCAB_FILE.exists() else 'seed vocabulary'}")
    
    print("\n[2] Calculating LCR/LOR for each model...")
    results = {}
    for model_name, outputs in model_outputs.items():
        print(f"\n    Processing {model_name}...")
        lcr_lor = calculate_lcr_lor(outputs, refs, cultural_vocab)
        results[model_name] = lcr_lor
        print(f"      - Total images: {lcr_lor['total_images']}")
        print(f"      - Cultural references: {lcr_lor['total_cultural_refs']}")
        print(f"      - Cultural matches: {lcr_lor['cultural_matches']}")
        print(f"      - LCR: {lcr_lor['lcr']:.4f}")
    
    # Save results
    output_dir = TASK_ROOT / "output" / "task5"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "lcr_lor_metrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n[3] Results saved to: {output_file}")
    
    print(f"\nNote: These metrics use train-only V_cultural when {VOCAB_FILE.name} exists.")

if __name__ == '__main__':
    main()
