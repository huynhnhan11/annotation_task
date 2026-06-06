#!/usr/bin/env python3
"""
Task 0 Step 2 (Train Only): Cluster vocabulary candidates using PhoBERT
- Load candidates from Step 1 (train-only version)
- Encode using PhoBERT embeddings
- Cluster semantically similar phrases
- Output grouped candidates for easier review
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import warnings

warnings.filterwarnings('ignore')

TASK_ROOT = Path(__file__).resolve().parents[2]

def load_candidates():
    """Load candidates from Step 1 (train-only version)"""
    with open(TASK_ROOT / "output" / "task0" / "task0_candidates_train_only.json", 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    
    phrases = list(candidates.keys())
    return phrases, candidates

def encode_with_phobert(phrases):
    """
    Encode phrases using PhoBERT embeddings.
    PhoBERT uses RDRSegmenter for tokenization.
    """
    print("    Initializing PhoBERT transformer...")
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("    ERROR: transformers or torch not installed")
        print("    Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', '-q', 'transformers', 'torch', 'scikit-learn'])
        from transformers import AutoTokenizer, AutoModel
        import torch
    
    # Load PhoBERT (pre-trained on Vietnamese)
    model_name = "vinai/phobert-base"
    print(f"    Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    embeddings = []
    batch_size = 32
    
    print(f"\n    Encoding {len(phrases)} phrases...")
    with torch.no_grad():
        for i in range(0, len(phrases), batch_size):
            batch = phrases[i:i+batch_size]
            if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(phrases):
                print(f"      [{i + len(batch)}/{len(phrases)}]")
            
            # PhoBERT expects space-separated tokens
            # Our phrases already use underscores for syllables, so they're pre-segmented
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            outputs = model(**inputs)
            
            # Use [CLS] token embedding as phrase representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embeddings)
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    print(f"    ✓ Embeddings shape: {embeddings.shape}")
    
    return embeddings, tokenizer, model

def cluster_phrases(embeddings, phrases, distance_threshold=0.3):
    """
    Cluster phrases using hierarchical clustering based on cosine distance.
    """
    print(f"\n    Computing distances...")
    distances = cosine_distances(embeddings)
    
    print(f"    Clustering with distance_threshold={distance_threshold}...")
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average',
        metric='precomputed'
    )
    
    labels = clusterer.fit_predict(distances)
    n_clusters_found = len(set(labels))
    print(f"    ✓ Found {n_clusters_found} clusters")
    
    # Group phrases by cluster
    clusters = defaultdict(list)
    for phrase, label in zip(phrases, labels):
        clusters[label].append(phrase)
    
    return clusters

def save_clustered_candidates(clusters, candidates):
    """Save clustered candidates to JSON and readable text format"""
    output_dir = TASK_ROOT / "output" / "task0"
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    # Save as JSON with cluster info
    clustered_json = {}
    for cluster_id, (orig_id, phrases) in enumerate(sorted_clusters):
        cluster_info = {
            "cluster_id": cluster_id,
            "size": len(phrases),
            "phrases": [
                {
                    "phrase": p,
                    "frequency": candidates[p]["frequency"],
                    "words": candidates[p].get("words", 0)
                }
                for p in sorted(phrases, key=lambda p: candidates[p]["frequency"], reverse=True)
            ]
        }
        clustered_json[f"cluster_{cluster_id}"] = cluster_info
    
    output_json = output_dir / "task0_clustered_candidates_train_only.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(clustered_json, f, ensure_ascii=False, indent=2)
    
    # Save as readable text for human review
    output_txt = output_dir / "task0_clustered_candidates_train_only.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CLUSTERED VOCABULARY CANDIDATES (TRAIN DATA ONLY)\n")
        f.write("For annotation: mark each as cultural (Y/N) and assign facet\n")
        f.write("=" * 80 + "\n\n")
        
        for cluster_id, (orig_id, phrases) in enumerate(sorted_clusters):
            # Representative phrase (highest frequency)
            rep_phrase = max(phrases, key=lambda p: candidates[p]["frequency"])
            f.write(f"\n[CLUSTER {cluster_id}] ({len(phrases)} phrases)\n")
            f.write(f"  Representative: {rep_phrase}\n")
            f.write(f"  Members:\n")
            
            for phrase in sorted(phrases, key=lambda p: candidates[p]["frequency"], reverse=True):
                freq = candidates[phrase]["frequency"]
                f.write(f"    - {phrase:<35} (freq: {freq})\n")
            
            f.write("\n  ☐ Cultural (Y/N):    _____\n")
            f.write("  ☐ Facet:              _____\n")
            f.write("  ☐ Notes:              _____\n")
            f.write("-" * 80 + "\n")
    
    return output_json, output_txt

def main():
    print("\n=== Task 0 Step 2 (Train Only): Cluster Vocabulary Candidates ===\n")
    
    # Load candidates
    print("[1] Loading candidates from Step 1 (train-only)...")
    phrases, candidates = load_candidates()
    print(f"    ✓ Loaded {len(phrases)} candidates")
    
    # Encode with PhoBERT
    print("\n[2] Encoding phrases with PhoBERT...")
    embeddings, tokenizer, model = encode_with_phobert(phrases)
    
    # Cluster
    print("\n[3] Clustering semantically similar phrases...")
    clusters = cluster_phrases(embeddings, phrases, distance_threshold=0.3)
    
    # Save
    print("\n[4] Saving results...")
    json_file, txt_file = save_clustered_candidates(clusters, candidates)
    print(f"    ✓ {json_file}")
    print(f"    ✓ {txt_file}")
    
    # Summary
    print(f"\n[5] Summary:")
    print(f"    - Total candidates: {len(phrases)}")
    print(f"    - Clusters created: {len(clusters)}")
    print(f"    - Avg. cluster size: {len(phrases) / len(clusters):.1f}")
    
    cluster_sizes = sorted([len(cluster) for cluster in clusters.values()], reverse=True)
    print(f"    - Largest cluster: {cluster_sizes[0]} phrases")
    print(f"    - Smallest cluster: {cluster_sizes[-1]} phrases")
    
    print(f"\nNext: Human annotation (Task 0 Step 3-4)")

if __name__ == '__main__':
    main()
