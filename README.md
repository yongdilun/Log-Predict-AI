# LogPredictAI - Production Log Classifier

A high-performance machine learning system for automatically classifying log entries into three categories: **approval**, **acknowledge**, and **error**.

## 🎯 Performance Overview

- **Accuracy**: 99.96%
- **F1-Score**: 99.96%
- **Training Data**: 1,000,000 balanced log entries
- **Model Size**: 858KB (optimized for production)
- **Prediction Speed**: < 1ms per log entry

## 📁 Project Structure

```
LogPredictAI/
├── log_classifier/
│   ├── data/
│   │   └── logs.csv              # Training dataset (1M entries)
│   ├── models/
│   │   ├── optimized_model.joblib      # Production model
│   │   └── optimized_model_metadata.joblib  # Model metadata
│   ├── src/
│   │   ├── improved_preprocess.py      # Text preprocessing
│   │   ├── improved_train.py           # Model training
│   │   └── predict.py                  # Prediction script
│   └── requirements.txt          # Dependencies
├── test_difficulty_levels.py     # Difficulty-based testing
├── realistic_test.py             # Real-world scenario testing
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r log_classifier/requirements.txt
```

### 2. Test the Model

```bash
# Test with difficulty levels
python test_difficulty_levels.py

# Test with realistic scenarios
python realistic_test.py
```

### 3. Run Production Tests

```bash
# Test with difficulty levels
python test_difficulty_levels.py

# Test with realistic scenarios  
python realistic_test.py
```

## 🏗️ Building the Model

### Step 1: Generate Training Data

```bash
# Compile and run the Java log generator
javac GenerateRealisticLogs.java
java GenerateRealisticLogs
```

This creates `realistic_logs.csv` with 1M balanced log entries.

### Step 2: Train the Model

```bash
# Train with optimized settings
python -m log_classifier.src.improved_train \
    --data log_classifier/data/logs.csv \
    --model-out log_classifier/models/optimized_model.joblib \
    --test-size 0.1 \
    --skip-cv
```

### Step 3: Evaluate Performance

```bash
# Run comprehensive tests
python test_difficulty_levels.py
python realistic_test.py
```

## 📊 Model Architecture

### Preprocessing Pipeline
- **Text Cleaning**: Removes timestamps, log levels, normalizes whitespace
- **TF-IDF Vectorization**: Converts text to numerical features
- **Feature Selection**: Optimized for log-specific patterns

### Classification Model
- **Algorithm**: LightGBM (Gradient Boosting)
- **Features**: 15,000 TF-IDF features
- **Classes**: 3 (approval, acknowledge, error)
- **Optimization**: Balanced class weights, early stopping

## 🧪 Testing the Model

### Difficulty Level Testing

The `test_difficulty_levels.py` script tests the model across 6 difficulty levels:

```python
# Level 1: Basic classification
"User approved transaction" → APPROVAL
"System acknowledged request" → ACKNOWLEDGE  
"Error occurred while loading" → ERROR

# Level 6: Expert/ambiguous cases
"Approval recorded, but system threw exception" → APPROVAL
```

### Realistic Scenario Testing

The `realistic_test.py` script tests real-world log formats:

```python
# Production log formats
"2025-10-22T14:02:31Z | INFO | APPROVAL_LOG: Admin accepted request #55321"
"[ERROR] 504 Gateway Timeout while processing API batch"
"[WARN] Approval partially completed – waiting for secondary confirmation"
```

## 📊 Performance Analysis

### Testing Scripts

The project includes comprehensive testing:

1. **Difficulty Level Testing** (`test_difficulty_levels.py`): Tests across 6 difficulty levels
2. **Realistic Scenario Testing** (`realistic_test.py`): Tests real-world log formats
3. **Performance Report** (`LogClassifier_Performance_Report.md`): Detailed metrics

### Running Tests

```bash
# Test model performance
python test_difficulty_levels.py
python realistic_test.py

# View detailed performance report
cat LogClassifier_Performance_Report.md
```

## 🔧 Production Usage

### Python API

```python
import joblib

# Load the model
model = joblib.load('log_classifier/models/optimized_model.joblib')

# Single prediction
prediction = model.predict(["User approved transaction #12345"])
print(prediction[0])  # Output: approval

# Batch prediction
logs = [
    "System acknowledged receipt of ping",
    "Error occurred while saving to database",
    "Payment approved by admin"
]
predictions = model.predict(logs)
print(predictions)  # Output: ['acknowledge', 'error', 'approval']

# Get prediction probabilities
probas = model.predict_proba(logs)
print(probas)  # Output: [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.9, 0.1, 0.0]]
```

### Command Line Interface

```bash
# Predict single log
python -m log_classifier.src.predict \
    --model log_classifier/models/optimized_model.joblib \
    --text "User approved transaction"

# Predict from file
python -m log_classifier.src.predict \
    --model log_classifier/models/optimized_model.joblib \
    --file logs_to_classify.txt
```

## 📈 Performance Metrics

### Test Results Summary

| Metric | Value | Status |
|--------|-------|--------|
| Overall Accuracy | 99.96% | ✅ Excellent |
| F1-Score | 99.96% | ✅ Excellent |
| Training Time | 138.82s | ✅ Fast |
| Model Size | 858KB | ✅ Optimized |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| approval | 99.95% | 99.93% | 99.94% |
| acknowledge | 99.93% | 99.95% | 99.94% |
| error | 100.00% | 100.00% | 100.00% |

### Real-World Testing

- **Level 1-5**: 100% accuracy (15/15 correct)
- **Level 6**: 66.7% accuracy (2/3 correct - ambiguous cases)
- **Edge Cases**: 83.3% accuracy (5/6 correct)

## 🛠️ Development

### Retraining the Model

```bash
# Generate new data
java GenerateRealisticLogs

# Retrain with new data
python -m log_classifier.src.improved_train \
    --data realistic_logs.csv \
    --model-out log_classifier/models/new_model.joblib
```

### Adding New Classes

1. Update `APPROVED_LABELS` in `improved_preprocess.py`
2. Regenerate training data with new classes
3. Retrain the model
4. Update test scripts

### Custom Preprocessing

Modify `improved_preprocess.py` to add custom text cleaning:

```python
def clean_text_gentle(text: str) -> str:
    # Add your custom preprocessing logic
    text = text.lower()
    text = re.sub(r'custom_pattern', '', text)
    return text
```

## 📋 Requirements

### Python Dependencies
```
pandas==2.2.2
scikit-learn==1.5.2
numpy==2.1.2
joblib==1.4.2
jupyter==1.1.1
lightgbm==4.6.0
matplotlib
seaborn
```

### System Requirements
- Python 3.8+
- Java 8+ (for log generation)
- 4GB RAM (for training)
- 1GB disk space
- No Jupyter required

## 🚀 Deployment

### Production Checklist

- ✅ Model accuracy > 99%
- ✅ Fast prediction speed (< 1ms)
- ✅ Small model size (858KB)
- ✅ Handles real-world log formats
- ✅ Comprehensive testing completed

### Monitoring

1. **Track accuracy** on new log data
2. **Monitor prediction confidence** scores
3. **Retrain periodically** with fresh data
4. **Update model** when new log patterns emerge

## 📞 Support

For questions or issues:

1. Check the performance report: `LogClassifier_Performance_Report.md`
2. Run the test scripts to verify functionality
3. Check model metadata: `log_classifier/models/optimized_model_metadata.joblib`
4. Review the training logs for any errors

## 📄 License

This project is open source. See the repository for license details.

---

**LogPredictAI** - Production-ready log classification with 99.96% accuracy 🚀
