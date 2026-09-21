"""
Download the ACDC dataset (150 patients, ~2.5 GB) from the Human Heart Project Girder API.
No account needed. License: CC BY-NC-SA 4.0 — cite Bernard et al., IEEE TMI 2018.
Usage: python scripts/download_acdc.py --out data/raw/ACDC [--workers 6] [--skip-4d]
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
ACDC_COLLECTION = "637218c173e9f0047faa00fb"


def get_json(url, retries=5):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def list_folders(parent_id, parent_type):
    return get_json(f"{API}/folder?parentType={parent_type}&parentId={parent_id}&limit=500")


def list_items(folder_id):
    return get_json(f"{API}/item?folderId={folder_id}&limit=500")


def download_item(item, dest: Path, retries=5):
    if dest.exists() and dest.stat().st_size == item["size"]:
        return dest, 0
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{API}/item/{item['_id']}/download", timeout=600) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if dest.stat().st_size != item["size"]:
                raise IOError(f"size mismatch for {dest.name}")
            return dest, item["size"]
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def main(args):
    out = Path(args.out)
    db = [f for f in list_folders(ACDC_COLLECTION, "collection") if f["name"] == "database"][0]
    jobs = []
    for split in list_folders(db["_id"], "folder"):  # training / testing
        for patient in list_folders(split["_id"], "folder"):
            pdir = out / split["name"] / patient["name"]
            pdir.mkdir(parents=True, exist_ok=True)
            for item in list_items(patient["_id"]):
                if args.skip_4d and item["name"].endswith("_4d.nii.gz"):
                    continue
                jobs.append((item, pdir / item["name"]))
    total = sum(j[0]["size"] for j in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB -> {out}", flush=True)

    done = 0
    got = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download_item, it, d): d for it, d in jobs}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            got += futs[fut].stat().st_size
            if done % 50 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {got / 1e9:.2f} GB", flush=True)

    n_train = len(list((out / "training").glob("patient*")))
    n_test = len(list((out / "testing").glob("patient*")))
    n_4d = len(list(out.rglob("*_4d.nii.gz")))
    print(f"training={n_train} testing={n_test} 4d_volumes={n_4d}")
    if (n_train, n_test) != (100, 50):
        sys.exit("patient count mismatch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/ACDC")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-4d", action="store_true", help="only ED/ES frames + masks (~50 MB)")
    main(p.parse_args())
