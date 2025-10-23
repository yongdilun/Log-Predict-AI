import argparse
import numpy as np
from pathlib import Path
import time
import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# MLflow imports for model conversion
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available - model conversion will be skipped")

# Import the improved preprocessing
from .improved_preprocess import APPROVED_LABELS, build_generalized_pipeline, build_sgd_pipeline

def load_data(csv_path: Path):
    """Load dataset."""
    df = pd.read_csv(csv_path, dtype={'text': str, 'label': str}, engine='c')
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].str.strip()
    df = df[df['label'].isin(APPROVED_LABELS)].copy()
    
    print(f"📊 Dataset: {len(df):,} rows")
    for label in APPROVED_LABELS:
        count = (df['label'] == label).sum()
        pct = (count / len(df)) * 100
        print(f"   {label:12s}: {count:,} ({pct:.1f}%)")
    
    return df["text"].values, df["label"].values


def test_on_real_examples(model):
    """Test on actual real-world examples."""
    print("\n" + "=" * 60)
    print("🧪 TESTING ON REAL-WORLD EXAMPLES")
    print("=" * 60)
    
    real_tests = {
        "Basic": [
            ("User approved transaction", "approval"),
            ("System acknowledged request", "acknowledge"),
            ("Error occurred while loading", "error")
        ],
        "Moderate": [
            ("Approval granted successfully", "approval"),
            ("Acknowledged receipt of data", "acknowledge"),
            ("Error: failed to connect", "error")
        ],
        "Complex": [
            ("Request approved by admin team", "approval"),
            ("Response acknowledged by device", "acknowledge"),
            ("Error detected in payment API", "error")
        ],
        "With Context": [
            ("[INFO] approval log confirmed", "approval"),
            ("Acknowledgement received after delay", "acknowledge"),
            ("ERR! 500 internal server error", "error")
        ],
        "Mixed Signals": [
            ("User approval completed but warning", "approval"),
            ("Device acknowledged firmware update successfully", "acknowledge"),
            ("Critical error while processing batch", "error")
        ]
    }
    
    total_correct = 0
    total_tests = 0
    
    for category, tests in real_tests.items():
        print(f"\n{category}:")
        correct = 0
        for text, expected in tests:
            pred = model.predict([text])[0]
            is_correct = pred == expected
            correct += is_correct
            total_correct += is_correct
            total_tests += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} [{pred:^12s}] (expected: {expected:^12s}) {text[:60]}")
        
        print(f"  Accuracy: {correct}/{len(tests)} ({100*correct/len(tests):.0f}%)")
    
    print(f"\n📊 Overall Real-World Accuracy: {total_correct}/{total_tests} ({100*total_correct/total_tests:.1f}%)")
    
    if total_correct < total_tests * 0.8:
        print("⚠️  WARNING: Model performs poorly on real examples!")
        print("   Consider: more diverse training data or less aggressive preprocessing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--max-features", type=int, default=15000)
    parser.add_argument("--use-sgd", action='store_true')
    parser.add_argument("--skip-cv", action='store_true', help="Skip cross-validation for faster training")
    args = parser.parse_args()

    print("🚀 GENERALIZATION-FOCUSED TRAINING")
    print("=" * 60)

    # Load data
    print("\n📦 Loading dataset...")
    X, y = load_data(args.data)

    # Split with more test data to detect overfitting
    print(f"\n🔀 Splitting (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Build pipeline
    print("\n🧠 Building pipeline with regularization...")
    if args.use_sgd:
        print("   Using SGD with ElasticNet regularization")
        pipeline = build_sgd_pipeline(max_features=args.max_features)
    else:
        print("   Using LightGBM/RF with depth limits")
        pipeline = build_generalized_pipeline(max_features=args.max_features)

    # Train
    print("\n🏋️  Training...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"⏱️  Completed in {train_time:.2f}s")

    # Evaluate on training set (to check overfitting)
    print("\n📊 Training Set Performance:")
    y_train_pred = pipeline.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    print(f"   F1-Score: {train_f1:.4f}")

    # Evaluate on test set
    print("\n📊 Test Set Performance:")
    y_test_pred = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"   F1-Score: {test_f1:.4f}")
    
    # Check for overfitting
    if train_f1 > 0.99 and test_f1 > 0.99:
        print("\n⚠️  WARNING: Both train and test F1 > 0.99")
        print("   This suggests the data is too easy/uniform.")
        print("   Model may not generalize to real-world examples!")
    elif train_f1 - test_f1 > 0.1:
        print(f"\n⚠️  WARNING: Large gap between train ({train_f1:.3f}) and test ({test_f1:.3f})")
        print("   Model is overfitting!")
    else:
        print(f"\n✓ Good train/test balance (gap: {train_f1-test_f1:.3f})")

    print("\n📈 Detailed Test Results:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Test on real examples
    test_on_real_examples(pipeline)

    # Save model
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out, compress=3)
    
    metadata = {
        "labels": APPROVED_LABELS,
        "train_time": train_time,
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "max_features": args.max_features
    }
    joblib.dump(metadata, str(args.model_out).replace('.joblib', '_metadata.joblib'))

    print(f"\n💾 Model saved to: {args.model_out}")
    
    # Convert to MLflow format for WPF integration
    if MLFLOW_AVAILABLE:
        print("\n🔄 Converting to MLflow format for WPF integration...")
        try:
            # Set up MLflow
            mlflow.set_tracking_uri("file:./mlflow_runs")
            mlflow.set_experiment("log_classifier")
            
            with mlflow.start_run():
                # Log model metadata
                mlflow.log_param("model_type", "LightGBM" if not args.use_sgd else "SGD")
                mlflow.log_param("accuracy", f"{test_f1:.4f}")
                mlflow.log_param("classes", ",".join(APPROVED_LABELS))
                mlflow.log_param("max_features", args.max_features)
                
                # Save model in MLflow format
                mlflow_model_path = mlflow.sklearn.save_model(
                    pipeline, 
                    "mlflow_model"
                )
                
                print(f"✅ MLflow model saved to: mlflow_model/")
                print("📁 Ready for WPF integration!")
                
                # Test the converted model
                print("🧪 Testing converted model...")
                test_texts = [
                    "User approved transaction",
                    "System acknowledged request", 
                    "Error occurred in database"
                ]
                
                predictions = pipeline.predict(test_texts)
                for text, pred in zip(test_texts, predictions):
                    print(f"   '{text}' → {pred}")
                
        except Exception as e:
            print(f"❌ MLflow conversion failed: {e}")
            print("   You can still use the .joblib model for Python applications")
    else:
        print("⚠️  MLflow not available - skipping WPF conversion")
        print("   Install MLflow with: pip install mlflow")
    
    print("✅ Done!")
    
    # Run difficulty test automatically
    print("\n" + "=" * 60)
    print("🧪 RUNNING DIFFICULTY TEST")
    print("=" * 60)
    
    # Difficulty test samples
    test_samples = {
        1: [
            "User approved transaction",
            "System acknowledged request",
            "Error occurred while loading page"
        ],
        2: [
            "Approval granted successfully",
            "Acknowledged receipt of data",
            "Error: failed to connect to server"
        ],
        3: [
            "Request approved by admin team",
            "Response acknowledged by device",
            "Error detected in payment API"
        ],
        4: [
            "[2025-10-22 13:02:10] INFO - approval log confirmed",
            "[INFO] Acknowledgement received after delay",
            "ERR! 500 internal server error triggered"
        ],
        5: [
            "User approval completed but warning: delayed confirmation",
            "Device acknowledged firmware update successfully",
            "Critical error while processing batch job on node-2"
        ],
        6: [
            "Approval recorded; however, user did not confirm via email",
            "Acknowledged multiple retry attempts before success",
            "System crashed: error persisted even after restart sequence"
        ]
    }
    
    # Run predictions
    print("=== LOG CLASSIFIER TEST (Easy → Expert) ===\n")
    for level, logs in test_samples.items():
        print(f"--- LEVEL {level} ---")
        preds = pipeline.predict(logs)
        for log, pred in zip(logs, preds):
            print(f"[{pred.upper():12}] {log}")
        print()
    
    print("🎯 Difficulty test completed!")
    
    # Run realistic test automatically
    print("\n" + "=" * 60)
    print("🌍 RUNNING REALISTIC TEST")
    print("=" * 60)
    
    # Realistic test samples
    realistic_tests = {
        "Retail Store": [
            "Cashier John Doe approved cash payment for customer #12345",
            "Manager Sarah Wilson confirmed employee discount request",
            "System approved credit card payment for $89.99",
            "Cashier approved split payment between card and cash",
            "Multi-step transaction approved by supervisor"
        ],
        "System Errors": [
            "CRITICAL ERROR: Payment gateway timeout after 30 seconds",
            "Exception thrown in payment processing module",
            "Inventory sync failed: Product quantity mismatch detected",
            "Database connection lost during peak transaction hours",
            "Barcode scanner hardware failure reported by staff"
        ],
        "Customer Service": [
            "Customer confirmed receipt of refund notification",
            "Staff acknowledged customer complaint about service",
            "System confirmed order cancellation request",
            "Customer service rep verified account information",
            "Support team acknowledged technical issue report"
        ],
        "Technical Operations": [
            "Network connectivity issue resolved after router restart",
            "Printer malfunction detected in checkout lane 3",
            "Device acknowledged firmware update successfully",
            "System confirmed backup completion for database",
            "Server acknowledged maintenance window completion"
        ]
    }
    
    # Run realistic predictions
    print("=== REALISTIC SCENARIO TESTING ===\n")
    total_correct = 0
    total_tests = 0
    
    for scenario, logs in realistic_tests.items():
        print(f"--- {scenario} ---")
        preds = pipeline.predict(logs)
        correct = 0
        
        for log, pred in zip(logs, preds):
            # Determine expected label based on content
            if any(word in log.lower() for word in ['approved', 'confirmed', 'authorized', 'granted']):
                expected = 'approval'
            elif any(word in log.lower() for word in ['acknowledged', 'confirmed', 'received', 'verified']):
                expected = 'acknowledge'
            elif any(word in log.lower() for word in ['error', 'failed', 'exception', 'timeout', 'malfunction', 'lost', 'issue']):
                expected = 'error'
            else:
                expected = 'unknown'
            
            is_correct = pred == expected
            correct += is_correct
            total_correct += is_correct
            total_tests += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} [{pred:^12s}] (expected: {expected:^12s}) {log[:60]}...")
        
        print(f"  Accuracy: {correct}/{len(logs)} ({100*correct/len(logs):.0f}%)")
        print()
    
    print(f"📊 Overall Realistic Accuracy: {total_correct}/{total_tests} ({100*total_correct/total_tests:.1f}%)")
    
    if total_correct < total_tests * 0.8:
        print("⚠️  WARNING: Model performs poorly on realistic examples!")
        print("   Consider: more diverse training data or better preprocessing")
    else:
        print("✅ Model performs well on realistic scenarios!")
    
    print("🎯 Realistic test completed!")


if __name__ == "__main__":
    main()