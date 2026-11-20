# Anonymous artifact: Two Ledgers

This repository contains the code and the result files for the paper
"Two Ledgers: How Far Credentialed Colluders Can Move a Traffic-State
Estimate". It is anonymized for double-blind review.

## Contents

- `src/` — decision layer, three attack modes, conservation and spatial
  conflict checks, VeReMi NextGen loader, SUMO trace generator, and the
  experiment driver `overnight.py`.
- `*.py` — experiment drivers (`run_*.py`), the median and Kalman checks
  (`median_check.py`, `kf_observer.py`), the cluster bootstrap
  (`boot_ci.py`), and the figure scripts (`phase_fig.py`, `fig2_attrib.py`,
  `fig3_snr.py`, `fig_nmin.py`, `fig_model_v3.py`).
- `outputs/*.jsonl` — result files behind the tables, including
  `phase.jsonl`, `nmin_big_*.jsonl`, `qdet.jsonl`, `spacing.jsonl`,
  `sumo_v2.jsonl`, and the IDM shards.
- `outputs/series.json` — the 15 s window used in Figure 1(c).
- `outputs/fig3_snr.png` — Figure 4, regenerated with the two-sided
  99th-percentile calibration (`z = 2.576`) and `p_eff = 0.98 p`.
- `sumo/` — SUMO network, route, and configuration files for the
  simulator layer.

## Requirements

Python 3.10 with `numpy`, `torch`, and `matplotlib`. The experiments run on
CPU or a single GPU; no other services are required.

## Data

Download the VeReMi NextGen dataset from Zenodo
(DOI `10.5281/zenodo.19665762`) and place the two highway archives at

    data/InTAS_highway_2.zip
    data/InTAS_highway_7.zip

`src/nextgen_load.py` reads the nested archive layout described in the
dataset distribution. The raw dataset is not included here because of its
size; the result files needed to check the paper's numbers are included.

## Reproducing the results

1. `python run_phase.py` — the 6,480-configuration NextGen grid →
   `outputs/phase.jsonl`.
2. `python run_nmin_big.py 0` and `python run_nmin_big.py 1` — the
   minimum-support sweep → `outputs/nmin_big_0.jsonl`,
   `outputs/nmin_big_1.jsonl`.
3. `python run_qdet.py` — counter-accuracy sweep → `outputs/qdet.jsonl`.
4. `python run_spacing.py` — phantom-spacing sweep →
   `outputs/spacing.jsonl`.
5. `python median_check.py` — mean vs lower-median table.
6. `python kf_observer.py` — Kalman temporal-smoothing baseline.
7. `python boot_ci.py` — cluster-bootstrap intervals for the headline rates.
8. `python phase_fig.py`, `python fig2_attrib.py`, `python fig3_snr.py`,
   `python fig_nmin.py`, `python fig_model_v3.py` — the paper figures.

The `outputs/` directory already contains the result files used in the
paper, so the tables can be checked without rerunning the simulations.

## Revision note

This directory is the canonical artifact for the submission. The 2026-11-19
revision regenerated `fig3_snr.py` and `outputs/fig3_snr.png` with the
two-sided 99th-percentile calibration (`z = 2.576`) and `p_eff = 0.98 p`;
all other files are unchanged. When mirroring the artifact, make the
repository root match this directory.
