"""
Convert raw ACDC / M&Ms downloads into (R, s_R, s_t, I_t) pair cache.
Usage: python scripts/preprocess.py --data_root data/raw --out_dir data/processed
"""
import argparse


def main(args):
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
