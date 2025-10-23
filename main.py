import argparse
import json
import sys

from pydantic import ValidationError

from train import run_pipeline
from predict import run_prediction

from typedefs import Config, Mode


def main():
    parser = argparse.ArgumentParser(description="MedSAM Pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=[Mode.TRAIN.value, Mode.PREDICT.value], default=Mode.TRAIN.value, help="Training or prediction mode")
    parser.add_argument("--image_path", type=str, help="Path to the image for prediction (required in prediction mode)")
    parser.add_argument("--checkpoint_path", type=str, help="Path to SAM model checkpoint for prediction (required in prediction mode)")
    parser.add_argument("--use-augmentations", action="store_true", help="Whether to use data augmentations during training")
    args = parser.parse_args()

    print(f"[INFO] Hello from medsam-pipeline! Using config from: {args.config}")

    try:
        with open(args.config, 'r') as f:
            config = Config.model_validate_json(f.read())

    except Exception as e:
        if isinstance(e, ValidationError):
            for item in json.loads(e.json()):
                if item["type"] == "missing":
                    print(f"[ERROR] Missing config setting in {args.config}: {item.get('loc')}")
                else:
                    print(f"[ERROR] Invalid config setting in {args.config}: {item.get('loc')} {item.get('msg')}")
            sys.exit(1)

    print("Configuration:")
    for item in config.model_dump().items():
        print(f"  {item[0]}: {item[1]}")

    if args.mode == Mode.TRAIN.value:
        print("[INFO] Running in training mode...")
        run_pipeline(config)

    elif args.mode == Mode.PREDICT.value:
        if not args.image_path:
            print("[ERROR] --image_path is required in prediction mode")
            sys.exit(1)
        if not args.checkpoint_path:
            print("[ERROR] --checkpoint_path is required in prediction mode")
            sys.exit(1)
        print("[INFO] Running in prediction mode...")
        run_prediction(config, args.image_path, args.checkpoint_path)

if __name__ == "__main__":
    main()
    