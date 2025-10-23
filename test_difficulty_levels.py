import joblib
from pathlib import Path

# Load model
model_path = Path("log_classifier/models/optimized_model.joblib")
model = joblib.load(model_path)

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
    preds = model.predict(logs)
    for log, pred in zip(logs, preds):
        print(f"[{pred.upper():12}] {log}")
    print()
