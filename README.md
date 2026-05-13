# AI Regulation Games

Simulation code, saved outputs, and figures for the paper
`paper/backfiring-paper.pdf`.

## Repository Layout

- `scripts/`: simulation and utility scripts.
- `data/simulation_outputs/`: saved CSV simulation outputs used by the figures.
- `figures/paper/`: selected final paper figure artifacts.
- `figures/archive/`: retained exploratory or earlier figure artifacts.
- `notebooks/archive/`: original exploratory notebooks retained for provenance.
- `notebooks/paper_reproduction/`: lean public-facing reproduction notes.
- `paper/`: paper PDF.
- `docs/`: publication and maintenance notes.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Released Figure Artifacts

The released paper figures are already saved in `figures/paper/`. To collect
them into a disposable output directory without changing the originals:

```sh
python scripts/collect_paper_figures.py
```

This writes copies to `outputs/paper_figures/`. It does not rerun simulations.

## Running New Simulations

The simulation script writes new relative outputs to `data/simulation_outputs/`:

```sh
python scripts/parallelnobargainFTGscript.py
```

Use this only for intentional new experiment generation. The existing CSV files
are saved outputs from prior runs.

