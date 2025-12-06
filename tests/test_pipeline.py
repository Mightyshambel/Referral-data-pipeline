import subprocess
import sys
import pandas as pd
from pathlib import Path

def test_pipeline_runs_and_outputs():
    # run the pipeline
    res = subprocess.run([sys.executable, "scripts/your_script.py", "--data_dir", "data", "--out_dir", "out"], check=False)
    # ensure process finished
    assert res.returncode == 0 or res.returncode is None

    out_file = Path("out/referral_report.csv")
    assert out_file.exists(), "Output referral_report.csv not found"

    df = pd.read_csv(out_file)
    # check row count expected
    assert len(df) == 46, f"Expected 46 rows, got {len(df)}"
