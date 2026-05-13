"""Collect the saved paper figures into a disposable output folder.

This script does not run simulations or regenerate plots. It copies the existing
figure files selected for the paper into ``outputs/paper_figures`` so a release
package can be assembled without touching the archived source figures.
"""

from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "figures" / "paper"
OUTPUT_DIR = REPO_ROOT / "outputs" / "paper_figures"

PAPER_FIGURES = [
    "teaser_figure_oct16_manip.png",
    "teaser_figure_oct16_manip_generalist.png",
    "teaser_figure_fine_tuning_regulation.png",
    "teaser_figure_june9.png",
    "phase_diagrams_with_bargaining_june9.png",
    "multivariate_analysis_pareto_regulation_improves_june10.png",
    "separable_highd_fourpanel.png",
    "positive_crossterms_fourpanel_june9.png",
    "negative_crossterms_fourpanel_june9.png",
    "safety_investments_example_game_jun9.png",
    "all_attainable_utilities_nobargaining_crossterms_june9.png",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for filename in PAPER_FIGURES:
        source = FIGURE_DIR / filename
        if not source.exists():
            missing.append(filename)
            continue
        shutil.copy2(source, OUTPUT_DIR / filename)

    if missing:
        missing_list = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(f"Missing paper figure files:\n{missing_list}")

    print(f"Copied {len(PAPER_FIGURES)} figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

