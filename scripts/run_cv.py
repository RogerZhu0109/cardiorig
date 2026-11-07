"""
Train cross-validation runs in parallel on one GPU, syncing checkpoints to a private Hugging Face
repo so a reclaimed Colab VM can resume. Every minute it prints one status line with progress, ETA
and any failed or stalled run, which also keeps the kernel alive.
Usage: python scripts/run_cv.py --model rig --folds 0 1 2 3 4 --members 1 2 --cache /content/cache --features /content/cache/cinema
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "RZ0109/stacom-runs"
SYNC_SECONDS = 1200
STALL_SECONDS = 900
ITERS = dict(rig=20000, current=40000, krebs=150000, phaseflow=37500)


def command(args, fold: int, member: int, out: Path) -> list[str]:
    seed = 100 * member + fold
    common = ["--fold", str(fold), "--seed", str(seed), "--cache", args.cache, "--features", args.features, "--out", str(out)]
    if args.model in ("rig", "krebs", "phaseflow"):
        return [sys.executable, f"scripts/train_{args.model}.py", *common]
    return [sys.executable, "scripts/train.py", "--objective", "regress", "--augment", "--iters", "40000", *common]


def sync(api: HfApi, out: Path, folder: str) -> None:
    for f in out.glob("*.pt"):
        if f.name != "tmp.pt":
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{folder}/{out.name}/{f.name}", repo_id=REPO)


def latest(out: Path) -> tuple[int, float]:
    """Last logged iteration and speed of a run."""
    found = re.findall(r"it (\d+) .* ([\d.]+) it/s", (out / "log.txt").read_text())
    return (int(found[-1][0]), float(found[-1][1])) if found else (0, 0.0)


def status(procs: dict, seen: dict, total: int) -> str:
    """One status line, and the runs that failed or stopped moving."""
    now, finished, failed, stalled, iters, speeds = time.time(), 0, [], [], [], []
    for out, (p, _) in procs.items():
        it, speed = latest(out)
        code = p.poll()
        finished += code == 0
        if code:
            failed.append(out.name)
        if code is None:
            iters.append(it), speeds.append(speed)
            if it != seen.get(out, (None, 0))[0]:
                seen[out] = (it, now)
            elif now - seen[out][1] > STALL_SECONDS:
                stalled.append(out.name)
    left = (total - min(iters)) / (sum(speeds) / len(speeds)) / 60 if iters and sum(speeds) else float("nan")
    line = f"finished {finished}/{len(procs)} | slowest {min(iters, default=total)}/{total}"
    line += f" | {sum(speeds) / max(len(speeds), 1):.1f} it/s each | ETA {left:.0f} min"
    return line + "".join(f" | FAILED {n}" for n in failed) + "".join(f" | STALLED {n}" for n in stalled)


def main(args):
    api = HfApi()
    root = Path(args.root)
    names = [f"{args.model}_m{m}_f{k}" for m in args.members for k in args.folds]
    snapshot_download(REPO, allow_patterns=[f"{args.folder}/{n}/*" for n in names], local_dir=root)
    procs, env = {}, dict(os.environ, OMP_NUM_THREADS="2")
    for m in args.members:
        for fold in args.folds:
            out = root / args.folder / f"{args.model}_m{m}_f{fold}"
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / "log.txt", "a")
            procs[out] = (subprocess.Popen(command(args, fold, m, out), stdout=log, stderr=subprocess.STDOUT, env=env), log)
    seen, last_sync = {}, time.time()
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(60)
        print(status(procs, seen, ITERS[args.model]), flush=True)
        if time.time() - last_sync > SYNC_SECONDS:
            for out in procs:
                sync(api, out, args.folder)
            last_sync = time.time()
    for out in procs:
        sync(api, out, args.folder)
    failed = [out.name for out, (p, _) in procs.items() if p.returncode]
    for out in procs:
        if out.name in failed:
            print(f"FAILED {out.name}:", (out / "log.txt").read_text()[-800:], flush=True)
    print("FAILED" if failed else "ALL DONE", flush=True)
    sys.exit(bool(failed))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(ITERS), required=True)
    p.add_argument("--folds", type=int, nargs="+", required=True)
    p.add_argument("--members", type=int, nargs="+", required=True, help="seed = 100 * member + fold")
    p.add_argument("--cache", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--folder", default="cv_ensemble", help="folder in the Hugging Face repo")
    p.add_argument("--root", default="/content/runs")
    main(p.parse_args())
