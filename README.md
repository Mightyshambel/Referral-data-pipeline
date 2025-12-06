# Referral-data-pipeline

## Quick start (local)
1. Put input CSVs into `data/`
2. Install deps: `python -m pip install -r requirements.txt`
3. Run: `python scripts/your_script.py --data_dir data --out_dir out`
4. Outputs: `out/referral_report.csv`, `out/profiling_report.csv`, `out/diagnostic_report.csv`

## Docker
Build and run:
```bash
docker build -t referral-pipeline:latest .
docker run --rm -v "$(pwd)/out":/out referral-pipeline:latest

