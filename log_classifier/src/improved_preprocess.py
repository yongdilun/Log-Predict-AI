import re
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

APPROVED_LABELS: List[str] = ["approval", "acknowledge", "error"]


def clean_text_gentle(text: str) -> str:
    """Gentler cleaning that preserves more semantic information."""
    if not text:
        return ""
    
    # Keep important markers but normalize
    text = text.lower()
    
    # Remove only timestamps, keep log levels as they're informative
    text = re.sub(r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]', '', text)
    
    # Normalize multiple spaces
    text = ' '.join(text.split())
    
    return text


def build_better_vectorizer(max_features: int = 15000) -> TfidfVectorizer:
    """Vectorizer that captures more semantic meaning."""
    return TfidfVectorizer(
        preprocessor=clean_text_gentle,
        ngram_range=(1, 3),  # Include trigrams for better context
        min_df=2,            # Keep rare words
        max_df=0.95,
        max_features=max_features,
        stop_words=None,     # DON'T remove stop words - they carry meaning
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        dtype=np.float32,
        sublinear_tf=True,   # Use log scaling for better feature distribution
        token_pattern=r'\b\w+\b'  # Include single letters
    )


def build_lightgbm_classifier() -> 'lgb.LGBMClassifier':
    """LightGBM with better regularization."""
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed")
    
    return lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=8,          # Reduced depth to prevent overfitting
        learning_rate=0.05,   # Slower learning for better generalization
        num_leaves=31,
        min_child_samples=50, # Increased for regularization
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,        # L1 regularization
        reg_lambda=0.1,       # L2 regularization
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )


def build_random_forest_classifier() -> RandomForestClassifier:
    """RandomForest with better regularization."""
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=12,         # Limit depth
        min_samples_split=10, # More conservative splitting
        min_samples_leaf=4,   # Larger leaves
        max_features='sqrt',  # Use sqrt for better generalization
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )


def build_generalized_pipeline(max_features: int = 15000) -> Pipeline:
    """Pipeline optimized for generalization, not memorization."""
    if HAS_LIGHTGBM:
        clf = build_lightgbm_classifier()
    else:
        clf = build_random_forest_classifier()
    
    return Pipeline([
        ('tfidf', build_better_vectorizer(max_features=max_features)),
        ('clf', clf)
    ])


def build_sgd_pipeline(max_features: int = 15000) -> Pipeline:
    """SGD pipeline with regularization."""
    return Pipeline([
        ('tfidf', build_better_vectorizer(max_features=max_features)),
        ('clf', SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',  # Combined L1+L2
            alpha=0.001,           # Regularization strength
            l1_ratio=0.15,
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            early_stopping=True,
            validation_fraction=0.1,
            tol=1e-4
        ))
    ])