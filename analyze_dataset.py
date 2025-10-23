import pandas as pd
import numpy as np
from collections import Counter

def analyze_dataset():
    df = pd.read_csv('log_classifier/data/logs.csv')
    
    print("=== DATASET ANALYSIS ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n=== LABEL DISTRIBUTION ===")
    label_counts = df['label'].value_counts()
    print(label_counts)
    print(f"Class balance ratio: {label_counts.min() / label_counts.max():.3f}")
    
    print("\n=== SAMPLE TEXTS BY LABEL ===")
    for label in df['label'].unique():
        if pd.isna(label):
            continue
        print(f"\n--- {label.upper()} ({label_counts[label]} samples) ---")
        samples = df[df['label'] == label]['text'].head(3).tolist()
        for i, sample in enumerate(samples, 1):
            print(f"{i}. {sample[:100]}...")
    
    print("\n=== TEXT LENGTH ANALYSIS ===")
    df['text_length'] = df['text'].str.len()
    print(f"Average text length: {df['text_length'].mean():.1f}")
    print(f"Min length: {df['text_length'].min()}")
    print(f"Max length: {df['text_length'].max()}")
    
    print("\n=== COMMON WORDS BY LABEL ===")
    for label in df['label'].unique():
        if pd.isna(label):
            continue
        texts = df[df['label'] == label]['text'].str.lower()
        words = ' '.join(texts).split()
        word_counts = Counter(words)
        print(f"\n{label.upper()} - Top 10 words:")
        for word, count in word_counts.most_common(10):
            print(f"  {word}: {count}")

if __name__ == "__main__":
    analyze_dataset()
