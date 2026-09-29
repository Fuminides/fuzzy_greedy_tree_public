from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_results_assets import shift_assets


def test_shift_assets_write_figure_summary_and_macros(tmp_path):
    rows = []
    for seed in range(3):
        for shift in (0.0, 1.0):
            rows.append(
                {
                    "ds": "iris",
                    "seed": seed,
                    "k": shift,
                    "acc": 0.9 - 0.2 * shift,
                    "ds_cov": 0.91 - 0.05 * shift,
                    "ds_size": 1.2 + 1.8 * shift,
                    "gl_cov": 0.91 - 0.3 * shift,
                    "gl_size": 1.1,
                }
            )
    input_path = tmp_path / "shift.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    selected = shift_assets(
        input_path=input_path,
        summary_shift=1.0,
        generated_dir=tmp_path / "generated",
        figure_dir=tmp_path / "figures",
        stats_dir=tmp_path / "stats",
    )

    assert selected.ds_cov == 0.86
    assert (tmp_path / "figures" / "shift_coverage.pdf").stat().st_size > 0
    assert (tmp_path / "stats" / "shift_summary.csv").is_file()
    macros = (tmp_path / "generated" / "shift_summary.tex").read_text()
    assert r"\ShiftLevel}{1}" in macros
    assert r"\ShiftConformalCoverage}{61.0\%}" in macros
