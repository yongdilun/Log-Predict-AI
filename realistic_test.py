import joblib
from pathlib import Path

# Load model
model_path = Path("log_classifier/models/optimized_model.joblib")
model = joblib.load(model_path)

# More Realistic Real-World Test Samples
test_samples = {
    1: [  # Easy: Clean and direct
        "Transaction approved successfully",
        "System acknowledged receipt of ping",
        "Error while saving record to database"
    ],
    2: [  # Moderate: Includes metadata and abbreviations
        "[INFO] Approval granted by manager ID#432",
        "[OK] ACK received for POST /api/login",
        "[ERROR] Disk read failure on node A-3"
    ],
    3: [  # Harder: Mixed signals, longer sentences
        "User approval completed after timeout warning",
        "Device sent acknowledgement after retry sequence",
        "Unexpected error during payment callback handling"
    ],
    4: [  # Real system logs
        "2025-10-22T14:02:31Z | INFO | APPROVAL_LOG: Admin accepted request #55321",
        "2025-10-22T14:03:14Z | DEBUG | ACK response from POS terminal (code=200)",
        "2025-10-22T14:04:01Z | ERROR | 504 Gateway Timeout while processing API batch"
    ],
    5: [  # Complex mixed cases
        "[WARN] Approval partially completed – waiting for secondary confirmation",
        "[TRACE] Firmware update acknowledged; verification pending",
        "[CRITICAL] Fatal error: cannot recover application context"
    ],
    6: [  # Expert: Ambiguous / noisy / multiple events
        "Approval recorded, but system threw non-critical exception (status: OK)",
        "Ack flag set true; however, confirmation email not sent",
        "Multiple cascading errors occurred; database rollback initiated"
    ]
}

# Run predictions
print("=== REALISTIC LOG CLASSIFIER TEST ===")
print("Testing model on real-world log scenarios\n")

for level, logs in test_samples.items():
    print(f"--- LEVEL {level} (Realistic Scenarios) ---")
    preds = model.predict(logs)
    
    for i, (log, pred) in enumerate(zip(logs, preds)):
        # Add some visual indicators for expected vs actual
        expected = ["approval", "acknowledge", "error"][i % 3]
        status = "✓" if pred == expected else "✗"
        
        print(f"[{pred.upper():^12}] {status} {log}")
        if pred != expected:
            print(f"             Expected: {expected.upper()}")
    
    print()

# Additional edge cases
print("--- EDGE CASES ---")
edge_cases = [
    "Approval workflow completed successfully",
    "Message acknowledged by recipient",
    "Database connection error occurred",
    "User approved the transaction",
    "System acknowledged the request",
    "Critical error in payment processing"
]

edge_preds = model.predict(edge_cases)
for log, pred in zip(edge_cases, edge_preds):
    print(f"[{pred.upper():^12}] {log}")

print("\n=== TEST COMPLETE ===")
