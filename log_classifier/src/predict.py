import argparse
from pathlib import Path
from typing import List

import joblib
import pandas as pd


def predict_texts(model_path: Path, texts: List[str]) -> pd.DataFrame:
	pipeline = joblib.load(model_path)
	# Try to load metadata separately
	metadata_path = str(model_path).replace('.joblib', '_metadata.joblib')
	labels = None
	try:
		metadata = joblib.load(metadata_path)
		labels = metadata.get("labels")
	except Exception:
		labels = ["approval", "acknowledge", "error"]  # fallback
	
	preds = pipeline.predict(texts)
	probas = None
	if hasattr(pipeline, "predict_proba"):
		try:
			probas = pipeline.predict_proba(texts)
		except Exception:
			probas = None
	data = {"text": texts, "pred": preds}
	if probas is not None and labels is not None:
		# Handle case where model might not have all classes in training
		unique_labels = pipeline.classes_ if hasattr(pipeline, 'classes_') else labels
		for i, label in enumerate(unique_labels):
			if i < probas.shape[1]:  # Check bounds
				data[f"p_{label}"] = probas[:, i]
	return pd.DataFrame(data)


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict classes for logs")
	parser.add_argument(
		"--model",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "models" / "log_model.joblib",
		help="Path to trained model file",
	)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--text", nargs="+", help="One or more log lines to classify")
	group.add_argument("--file", type=Path, help="Path to a text file with one log per line")
	args = parser.parse_args()

	texts: List[str]
	if args.text:
		texts = args.text
	else:
		with args.file.open("r", encoding="utf-8") as f:
			texts = [line.rstrip("\n") for line in f]

	df = predict_texts(args.model, texts)
	print(df.to_csv(index=False))


if __name__ == "__main__":
	main()