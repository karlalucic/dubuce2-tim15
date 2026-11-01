"""Download and preprocess the Kenneth French 10 Industry Portfolios (daily) data.

Downloads the raw CSV zip into data/raw/, parses only the
"Average Value Weighted Returns -- Daily" section, converts percent returns
to decimal returns, keeps 2000-01-01 to 2024-12-31 and writes
data/processed/industry10_daily_2000_2024.csv.

Usage:
    python scripts/download_data.py
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "10_Industry_Portfolios_daily_CSV.zip"
)
ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "10_Industry_Portfolios_daily_CSV.zip"
PROCESSED_CSV = ROOT / "data" / "processed" / "industry10_daily_2000_2024.csv"

SECTION_TITLE = "Average Value Weighted Returns -- Daily"
COLUMNS = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
           "Telcm", "Shops", "Hlth", "Utils", "Other"]
MISSING_CODES = [-99.99, -999.0]
START, END = "2000-01-01", "2024-12-31"


def download(url: str = URL, dest: Path = RAW_ZIP) -> Path:
    """Download the raw zip to dest (skipped if it already exists)."""
    if dest.exists():
        print(f"Using cached {dest.relative_to(ROOT)}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Downloaded {dest.relative_to(ROOT)} ({len(response.content):,} bytes)")
    return dest


def read_csv_text(zip_path: Path) -> str:
    """Return the text of the single CSV file inside the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {zip_path}, found {names}")
        return zf.read(names[0]).decode("latin-1")


def parse_value_weighted(text: str) -> pd.DataFrame:
    """Parse the value-weighted daily section into a DataFrame of decimal returns.

    The section starts after the line containing SECTION_TITLE, has one header
    line, and ends at the first blank line.
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == SECTION_TITLE)
    header_idx = start + 1
    end = header_idx + 1
    while end < len(lines) and lines[end].strip():
        end += 1

    section = "\n".join(lines[header_idx:end])
    df = pd.read_csv(io.StringIO(section), index_col=0, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    if list(df.columns) != COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.index.name = "date"
    df = df.astype(float).replace(MISSING_CODES, np.nan)
    return df / 100.0


def main() -> None:
    zip_path = download()
    returns = parse_value_weighted(read_csv_text(zip_path))
    returns = returns.loc[START:END]

    assert not returns.isna().any().any(), "Missing values in the 2000-2024 range"
    assert returns.shape[1] == 10

    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(PROCESSED_CSV, float_format="%.6f")
    print(
        f"Wrote {PROCESSED_CSV.relative_to(ROOT)}: {len(returns)} rows x "
        f"{returns.shape[1]} columns, {returns.index[0].date()} to "
        f"{returns.index[-1].date()}"
    )


if __name__ == "__main__":
    main()
