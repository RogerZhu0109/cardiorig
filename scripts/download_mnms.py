"""
Download the M&Ms open dataset (Campello et al., IEEE TMI 2021; 345 subjects, 15.4 GB) from its
public MEGA folder: OpenDataset/{Training/Labeled,Training/Unlabeled,Validation,Testing}/CODE/
CODE_sa.nii.gz + CODE_sa_gt.nii.gz and the information CSV. The link is given on registration
(ub.edu/mnms) and is not kept in the repository. Resumable; every .nii.gz is read through once
(gzip CRC). MEGA limits anonymous transfer per IP: on "quota" re-run later or from another machine.
Needs pycryptodome.
Usage: python scripts/download_mnms.py --link 'https://mega.nz/folder/ID#KEY' --out data/raw/MnMs [--workers 4]
"""
import argparse
import base64
import gzip
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from Crypto.Cipher import AES
from Crypto.Util import Counter

API = "https://g.api.mega.co.nz/cs"


def b64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def api(folder: str, request: dict, retries: int = 8):
    for i in range(retries):
        req = urllib.request.Request(f"{API}?id={int(time.time() * 1000) % 10**9}&n={folder}",
                                     data=json.dumps([request]).encode(), headers={"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=60))
        out = out[0] if isinstance(out, list) else out
        if out == -17:
            sys.exit("quota: MEGA transfer limit reached for this IP")
        if not isinstance(out, int):
            return out
        time.sleep(2 ** i)  # -3 (try again) and friends
    sys.exit(f"MEGA API error {out}")


def list_files(folder: str, key: bytes) -> list[dict]:
    """Every file of the folder: path, size, handle, AES key and counter start."""
    info = {}
    for n in api(folder, {"a": "f", "c": 1, "r": 1})["f"]:
        k = AES.new(key, AES.MODE_ECB).decrypt(b64(n["k"].split(":")[1]))
        iv = None
        if n["t"] == 0:
            a = struct.unpack(">8I", k)
            k, iv = struct.pack(">4I", a[0] ^ a[4], a[1] ^ a[5], a[2] ^ a[6], a[3] ^ a[7]), (a[4] << 32) + a[5]
        attr = AES.new(k, AES.MODE_CBC, b"\0" * 16).decrypt(b64(n["a"]))
        info[n["h"]] = dict(name=json.loads(attr[4:].rstrip(b"\0"))["n"], parent=n["p"], size=n.get("s", 0),
                            file=n["t"] == 0, handle=n["h"], key=k, iv=iv)
    for f in info.values():
        parts, h = [], f["handle"]
        while h in info:
            parts.append(info[h]["name"])
            h = info[h]["parent"]
        f["path"] = "/".join(reversed(parts[:-1]))  # without the shared folder's own name
    return sorted((f for f in info.values() if f["file"]), key=lambda f: f["path"])


def download(folder: str, f: dict, dest: Path, retries: int = 5) -> int:
    if dest.exists() and dest.stat().st_size == f["size"]:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for i in range(retries):
        try:
            url = api(folder, {"a": "g", "g": 1, "n": f["handle"]})["g"]
            aes = AES.new(f["key"], AES.MODE_CTR, counter=Counter.new(128, initial_value=f["iv"] << 64))
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(aes.decrypt(chunk))
            if part.stat().st_size != f["size"]:
                raise IOError(f"size {part.stat().st_size} != {f['size']}")
            if dest.name.endswith(".gz"):
                with gzip.open(part) as g:
                    while g.read(1 << 24):
                        pass
            part.rename(dest)
            return f["size"]
        except urllib.error.HTTPError as e:
            if e.code == 509:
                raise SystemExit("quota: MEGA transfer limit reached for this IP")
            err = e
        except (OSError, EOFError) as e:
            err = e
        time.sleep(2 ** i)
    raise IOError(f"{f['path']}: {err}")


def main(args):
    folder, key = args.link.split("/folder/")[1].split("#")
    files, out = list_files(folder, b64(key)), Path(args.out)
    print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB -> {out}", flush=True)
    done = got = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(download, folder, f, out / f["path"]) for f in files]
        try:
            for fut in as_completed(futs):
                got += fut.result()
                done += 1
                if done % 20 == 0 or done == len(files):
                    print(f"[{done}/{len(files)}] {got / 1e9:.2f} GB new", flush=True)
        except SystemExit:
            ex.shutdown(cancel_futures=True)
            raise
    n_img, n_gt = len(list(out.rglob("*_sa.nii.gz"))), len(list(out.rglob("*_sa_gt.nii.gz")))
    print(f"cines={n_img} labels={n_gt} csv={len(list(out.rglob('*.csv')))}")
    if (n_img, n_gt) != (345, 345):
        sys.exit("count mismatch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--link", required=True)
    p.add_argument("--out", default="data/raw/MnMs")
    p.add_argument("--workers", type=int, default=4)
    main(p.parse_args())
