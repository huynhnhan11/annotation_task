#!/usr/bin/env python3
"""
Task 0 Step 1 (Train Only): Mine Vietnamese Cultural Vocabulary Candidates
- Load ONLY train_data.json (NOT test data)
- Extract noun phrases (space-separated, 1-3 words)
- Filter by frequency and remove generic phrases
- Output ~400-500 candidates
"""

import json
import re
from collections import Counter
import math
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TASK_ROOT.parent

# Function words to skip
FUNCTION_WORDS = {
    'là', 'và', 'ở', 'từ', 'trong', 'trên', 'dưới', 'ngoài', 'với', 
    'có', 'không', 'được', 'cái', 'chiếc', 'những', 'một', 'hai', 'ba',
    'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười', 'này', 'kia', 'đó',
    'đây', 'nào', 'nên', 'nếu', 'hay', 'hoặc', 'nhưng', 'tuy', 'mà', 'đến',
    'qua', 'theo', 'bằng', 'do', 'để', 'lại', 'còn', 'sẽ', 'đã', 'đang',
    'vừa', 'tại', 'trị', 'sao', 'gì', 'ai', 'nơi', 'thời', 'như', 'rồi',
    'mới', 'lên', 'xuống', 'ra', 'vào', 'cũng', 'thì', 'mặc', 'khoảng',
    'lúc', 'nay', 'chừng', 'thứ', 'chiều', 'phía', 'bên', 'góc', 'cạnh',
    'được', 'là_một', 'là_cái', 'bao_gồm', 'gồm', 'tối', 'sáng',
    'chính', 'riêng', 'khác', 'giống', 'tương_tự', 'cùng'
}

# Generic/non-cultural words to skip
GENERIC_PHRASES = {
    'hình_ảnh', 'khung_cảnh', 'cảnh_tượng', 'tình_cảnh', 'quang_cảnh',
    'cảnh', 'khung', 'hình', 'cách', 'điều', 'chuyện', 'vấn_đề',
    'khoảng', 'chừng', 'lúc', 'thời_điểm', 'lúc_này', 'lúc_đó'
}

def load_captions(data_file):
    """Load segmented captions"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions = []
    for ann in data.get('annotations', []):
        caption = ann.get('segment_caption') or ann.get('caption', '')
        if caption:
            captions.append(caption)
    
    return captions

def extract_noun_phrases(captions):
    """
    Extract noun phrases by:
    1. Split by SPACES (word boundaries, not underscores)
    2. Each token/word may contain underscores (within-word syllable breaks)
    3. Group consecutive non-function words (1-3 tokens)
    """
    all_phrases = []
    
    for sent in captions:
        if not sent:
            continue
        
        # Split by SPACES to get word tokens
        words = sent.split()
        
        # Extract consecutive non-function words (1-3 tokens)
        phrase = []
        for word in words:
            word_lower = word.lower()
            
            # If word is not a function word, add to phrase
            if word_lower not in FUNCTION_WORDS:
                phrase.append(word)
                
                # Save when reaching 3 tokens or at sentence end
                if len(phrase) == 3:
                    combined = ' '.join(phrase)
                    all_phrases.append(combined)
                    # Shift window: drop first word
                    phrase = phrase[1:]
            else:
                # Hit function word: save current phrase
                if 1 <= len(phrase) <= 3:
                    combined = ' '.join(phrase)
                    all_phrases.append(combined)
                phrase = []
        
        # Save remaining phrase
        if 1 <= len(phrase) <= 3:
            combined = ' '.join(phrase)
            all_phrases.append(combined)
    
    return all_phrases

def should_filter_out(phrase):
    """Check if phrase should be filtered out"""
    phrase_lower = phrase.lower()
    
    # Too generic
    if phrase_lower in GENERIC_PHRASES:
        return True
    
    # Contains function words or generic terms
    for word in phrase.split():
        if word.lower() in FUNCTION_WORDS or word.lower() in GENERIC_PHRASES:
            return True
    
    # Too short (single-char words)
    if any(len(w.replace('_', '')) <= 1 for w in phrase.split()):
        return True
    
    return False

def filter_candidates(phrases, min_freq=5):
    """
    Filter by:
    - Minimum frequency >= min_freq
    - Remove generic/function word phrases
    - Aim for 400-500 candidates
    """
    freq = Counter(phrases)
    
    candidates = {}
    for phrase, count in freq.most_common():
        if count >= min_freq and not should_filter_out(phrase):
            num_words = len(phrase.split())
            candidates[phrase] = {
                'frequency': count,
                'words': num_words
            }
            
            # Aim for ~400-500
            if len(candidates) >= 550:
                break
    
    return candidates

def main():
    print("\n=== Task 0 Step 1 (Train Only): Mine Vocabulary Candidates ===\n")
    
    # Load ONLY training captions
    print("[1] Loading TRAINING captions only...")
    train_captions = load_captions(REPO_ROOT / "data" / "train_data.json")
    
    print(f"    ✓ {len(train_captions)} training captions loaded")
    print(f"    (Note: NOT including test data)")
    
    # Extract phrases
    print("\n[2] Extracting noun phrases (space-separated, 1-3 words)...")
    phrases = extract_noun_phrases(train_captions)
    unique = set(phrases)
    print(f"    ✓ {len(phrases)} phrase instances")
    print(f"    ✓ {len(unique)} unique phrases")
    
    # Filter candidates
    print("\n[3] Filtering candidates (min_freq=5, remove generic)...")
    candidates = filter_candidates(phrases, min_freq=5)
    print(f"    ✓ Found {len(candidates)} candidates")
    
    # Sort by frequency (descending)
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda x: x[1]['frequency'],
        reverse=True
    )
    
    # Save results
    output_dir = TASK_ROOT / "output" / "task0"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    candidates_json = {phrase: info for phrase, info in sorted_candidates}
    output_file = output_dir / "task0_candidates_train_only.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(candidates_json, f, ensure_ascii=False, indent=2)
    
    # Save as CSV
    output_csv = output_dir / "task0_candidates_train_only.csv"
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("phrase,frequency,words\n")
        for phrase, info in sorted_candidates:
            f.write(f'"{phrase}",{info["frequency"]},{info["words"]}\n')
    
    print(f"\n[4] Saved results:")
    print(f"    ✓ {output_file}")
    print(f"    ✓ {output_csv}")
    
    # Stats
    word_counts = Counter(info['words'] for info in candidates.values())
    print(f"\n[5] Candidate statistics:")
    for num_words in sorted(word_counts.keys()):
        print(f"    - {num_words}-word phrases: {word_counts[num_words]}")
    
    # Print top 50
    print(f"\n[6] Top 50 candidates:")
    print("-" * 75)
    print(f"{'Rank':<5} {'Phrase':<45} {'Freq':>8} {'Words':>5}")
    print("-" * 75)
    for idx, (phrase, info) in enumerate(sorted_candidates[:50], 1):
        phrase_display = phrase if len(phrase) <= 45 else phrase[:42] + "..."
        print(f"{idx:<5} {phrase_display:<45} {info['frequency']:>8} {info['words']:>5}")
    print("-" * 75)
    
    print(f"\nNext step: Task 0 Step 2 - Cluster candidates semantically")

if __name__ == '__main__':
    main()
