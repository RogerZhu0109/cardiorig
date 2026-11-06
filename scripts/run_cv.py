"""
Train cross-validation folds in parallel on one GPU, syncing checkpoints to a private Hugging Face
repo so a reclaimed Colab VM can resume. Prints a heartbeat every minute, which keeps the kernel alive.
Usage: python scripts/run_cv.py --model rig --folds 0 1 2 3 4 --cache /content/cache --features /content/cache/cinema
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "RZ0109/stacom-runs"
SYNC_SECONDS = 1200


def command(args, fold: int, out: Path) -> list[str]:
    common = ["--fold", str(fold), "--seed", str(fold), "--cache", args.cache, "--features", args.features, "--out", str(out)]
    if args.model == "rig":
        return [sys.executable, "scripts/train_rig.py", *common]
    return [sys.executable, "scripts/train.py", "--objective", "regress", "--augment", "--iters", "40000", *common]


def sync(api: HfApi, out: Path) -> None:
    if (out / "last.pt").exists():
        api.upload_file(path_or_fileobj=str(out / "last.pt"), path_in_repo=f"cv/{out.name}/last.pt", repo_id=REPO)


def main(args):
    api = HfApi()
    root = Path(args.root)
    snapshot_download(REPO, allow_patterns=[f"cv/{args.model}_f{k}/*" for k in args.folds], local_dir=root.parent)
    procs = {}
    for fold in args.folds:
        out = root / f"{args.model}_f{fold}"
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / "log.txt", "a")
        procs[out] = (subprocess.Popen(command(args, fold, out), stdout=log, stderr=subprocess.STDOUT), log)
    last_sync = time.time()
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(60)
        tails = [(out.name, (out / "log.txt").read_text().strip().splitlines()[-1:]) for out in procs]
        print(time.strftime("%H:%M"), tails, flush=True)
        if time.time() - last_sync > SYNC_SECONDS:
            for out in procs:
                sync(api, out)
            last_sync = time.time()
    failed = [out.name for out, (p, _) in procs.items() if p.returncode]
    for out in procs:
        sync(api, out)
    print("failed:", failed or "none", flush=True)
    sys.exit(bool(failed))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["current", "rig"], required=True)
    p.add_argument("--folds", type=int, nargs="+", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--root", default="/content/runs/cv")
    main(p.parse_args())
