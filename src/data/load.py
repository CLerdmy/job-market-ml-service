from pathlib import Path
import polars as pl

def load_job_market_dataset() -> pl.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "job_market.csv"
    return pl.read_csv(data_path)