"""Run the full pipeline end to end: python run_all.py"""
import runpy
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

STEPS = [
    "01_ingest_and_simulate",
    "02_scorecard",
    "03_vintage_analysis",
    "04_roll_rates",
    "05_cutoff_profit",
    "06_limits_affordability",
    "07_ab_test",
    "08_pnl_forecast",
]

if __name__ == "__main__":
    for step in STEPS:
        t0 = time.time()
        print(f"\n=== {step} ===")
        runpy.run_path(str(SRC / f"{step}.py"), run_name="__main__")
        print(f"--- {step} done in {time.time() - t0:.1f}s")
