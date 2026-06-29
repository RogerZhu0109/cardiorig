"""
Run inference and compute metrics on the test set.
Usage: python scripts/evaluate.py --checkpoint checkpoints/best.pt
"""
import argparse


def main(args):
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    main(parser.parse_args())
