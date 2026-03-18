"""fastText training script for Semantic Denoising (Phase 8).

This script provides the scaffolding to train a fastText binary classifier
on real Kilo Code session logs.

Usage:
    python -m smartfork.indexer.fasttext_trainer --data metrics_labels.txt
"""

import sys
from pathlib import Path
from loguru import logger
import fasttext

def train_noise_classifier(data_path: Path, output_path: Path):
    """Train a fastText text classification model to detect terminal noise.
    
    Data format should be fastText standard:
    __label__NOISE npm ERR! code ERESOLVE...
    __label__CODE def my_function(): pass
    """
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}")
        logger.info("Please manually label 300-500 reasoning blocks with __label__NOISE or __label__CODE")
        return

    logger.info(f"Training fastText model on {data_path}...")
    
    # Train supervised model
    # WordNgrams=2 helps capture multi-word terminal phrases like 'npm error'
    model = fasttext.train_supervised(
        input=str(data_path),
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        bucket=200000,
        dim=50,
        loss='ova'
    )
    
    # Save the model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    logger.info(f"Model saved to {output_path}")

    # Optional: print test metrics if test data exists
    # results = model.test("test_data.txt")
    # print(f"Precision: {results[1]}, Recall: {results[2]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train fastText noise classifier")
    parser.add_argument("--data", type=str, default="data/labeled_noise.txt", help="Path to labeled training data")
    parser.add_argument("--out", type=str, default="models/noise_classifier.bin", help="Output path for the .bin model")
    
    args = parser.parse_args()
    
    train_noise_classifier(Path(args.data), Path(args.out))
