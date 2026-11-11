"""
Download the cine part of Multimodality STRAUS (18 virtual patients, ~1.1 GB) from the
Human Heart Project Girder API: cine images (mhd/raw), per-frame truth meshes (vtk) and
the 3 reference meshes (AHA labels, local directions). No account needed.
Usage: python scripts/download_straus.py --out data/raw/STRAUS [--workers 4]
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from download_acdc import download_item, list_folders, list_items

STRAUS_COLLECTION = "587de6f4e1af3f30a2980a58"


def main(args):
    out = Path(args.out)
    jobs = []
    for top in list_folders(STRAUS_COLLECTION, "collection"):
        if top["name"].startswith("reference_mesh"):
            out.mkdir(parents=True, exist_ok=True)
            jobs += [(item, out / item["name"]) for item in list_items(top["_id"])]
            continue
        cine = [f for f in list_folders(top["_id"], "folder") if f["name"] == "cine"][0]
        for sub in list_folders(cine["_id"], "folder"):  # image / mesh
            d = out / top["name"] / sub["name"]
            d.mkdir(parents=True, exist_ok=True)
            jobs += [(item, d / item["name"]) for item in list_items(sub["_id"])]
    total = sum(j[0]["size"] for j in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB -> {out}", flush=True)

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(download_item, it, d, 8, 30) for it, d in jobs]  # server hangs: short timeout
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}]", flush=True)

    n_pat = len(list(out.glob("patient*")))
    n_img = len(list(out.glob("patient*/image/*.raw")))
    n_mesh = len(list(out.glob("patient*/mesh/*.vtk")))
    print(f"patients={n_pat} frames={n_img} meshes={n_mesh}")
    if (n_pat, n_img, n_mesh) != (18, 540, 540):
        sys.exit("count mismatch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/STRAUS")
    p.add_argument("--workers", type=int, default=4)
    main(p.parse_args())
