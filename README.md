# AI Regulation Games

This is the repository corresponding to the paper entitled "The Backfiring
Effect of Weak AI Safety Regulation".

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The main simulation script is:

```sh
python scripts/parallelnobargainFTGscript.py
```

Saved simulation outputs are in `data/simulation_outputs/`, notebooks are in
`notebooks/archive/`, and figures are in `figures/`.
