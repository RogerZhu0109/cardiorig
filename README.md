<div align="center">

# CardioRig

### An Anatomical Rig for Predicting Cardiac Motion from a Single Cine MRI Frame

[![Project page](https://img.shields.io/badge/Project-Page-1f6feb)](https://rogerzhu0109.github.io/cardiorig/)

</div>

<p align="center">
  <img src="project%20page/assets/method.png" width="90%" alt="CardioRig overview">
</p>

CardioRig describes the motion of the whole cardiac cycle with 232 parameters, each tied to a named segment of the heart.
A network predicts them from one end-diastolic frame, and a fixed decoder turns them into a diffeomorphic deformation of that frame.
This repository contains the code for the paper: data preparation, training, evaluation and figures.

## Setup

Python 3.12, PyTorch 2.x.

```
pip install -r requirements.txt
```

The CineMA features are computed in a second environment (Python 3.11) with
`pip install git+https://github.com/mathpluscode/CineMA`, called `.venv-cinema` below.

## Data

```
python scripts/download_acdc.py                      # data/raw/ACDC
python scripts/preprocess.py                         # data/processed/acdc: slices and dense registration
.venv-cinema/bin/python scripts/cinema_features.py   # data/processed/acdc_cinema

python scripts/download_straus.py                    # data/raw/STRAUS
python scripts/preprocess_straus.py
.venv-cinema/bin/python scripts/cinema_features.py --cache data/processed/straus --out_dir data/processed/straus/cinema

python scripts/download_mnms.py --link <MEGA link received on registration>   # needs pycryptodome
python scripts/preprocess_mnms.py
.venv-cinema/bin/python scripts/cinema_features.py --cache data/processed/mnms --out_dir data/processed/mnms/cinema
```

## Training

Final models, five seeds, on the 100 ACDC training patients:

```
F=data/processed/acdc_cinema
for s in 0 1 2 3 4; do
  python scripts/train_rig.py       --fold -1 --seed $s --features $F --out checkpoints/final/rig_s$s
  python scripts/train.py           --fold -1 --seed $s --features $F --augment --iters 40000 --out checkpoints/final/current_s$s
  python scripts/train_phaseflow.py --fold -1 --seed $s --out checkpoints/final/phaseflow_s$s
  python scripts/train_krebs.py     --fold -1 --seed $s --out checkpoints/final/krebs_s$s
done
```

Cross-validation models use `--fold k --seed k` for k = 0..4 and `--out checkpoints/cv/{rig,current}_f$k`.
Ablations: `python scripts/run_cv.py --model rig --variant <variant> --folds 0 1 2 3 4 --members 0 1 2 --folder ablations --cache data/processed/acdc --features $F`
(variants are listed in `scripts/train_rig.py`; `full` needs members 1 and 2 only).

## Evaluation

```
python scripts/evaluate_test.py   --runs checkpoints/final --features $F                           # ACDC test set
python scripts/evaluate_mnms.py   --runs checkpoints/final --features data/processed/mnms/cinema   # M&Ms
python scripts/evaluate_straus.py --runs checkpoints/final                                         # STRAUS
python scripts/register_rig.py    --cache data/processed/acdc --straus data/processed/straus       # rig registration
python scripts/analyze_rig_size.py                                                                 # rig size
python scripts/analyze_readout.py --cache data/processed/acdc --fits outputs/rig_registration/fits # read-out
python scripts/evaluate_cv.py        --runs checkpoints/cv --features $F                           # cross-validation
python scripts/evaluate_ablations.py --runs checkpoints/ablations --full checkpoints/cv --features $F
python scripts/analyze_rig.py && python scripts/analyze_edits.py                                   # segment edits
```

Each script writes CSV files to `outputs/` and prints its tables. `figures/fig_*.py` draw the figures from these files.

## Names in the code

| Code | Paper |
| --- | --- |
| `rig` | CardioRig network |
| `current` | dense-field model |
| `teacher` | dense registration |
| `ceiling` | rig fitted to dense registration |
| `rig_registration` | rig fitted to the images |
| `identity` | no motion |

## Tests

```
python -m pytest tests
```

The header of each script gives its usage and outputs.
