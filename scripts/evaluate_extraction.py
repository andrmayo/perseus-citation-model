#!/usr/bin/env python3
"""Script to evaluate the citation extraction model on test set."""

import argparse
import logging

from perscit_model.extraction.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate citation extraction model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: use last trained model)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="model_data/extraction/test.jsonl",
        help="Path to test data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/extraction/test",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument(
        "--no-last-trained",
        action="store_true",
        help="Don't use last trained model (requires --model-path)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        last_trained=not args.no_last_trained,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
