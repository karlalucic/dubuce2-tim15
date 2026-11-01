# Portfolio Risk Estimation with a Diffusion Model

FER Deep Learning 2 (dubuce2), 2026/2027, Team 15.

We train a DDPM to generate daily returns for 10 US industry sectors and use the
generated scenarios to estimate portfolio VaR and ES. We compare it against a VAE,
historical simulation and a Ledoit-Wolf normal model, using walk-forward VaR/ES backtests.

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Data

The processed data is committed at `data/processed/industry10_daily_2000_2024.csv`,
so you don't need to download anything to get started. It holds daily decimal
returns (value-weighted) of the Kenneth French
[10 Industry Portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
2000-01-03 to 2024-12-31 (6289 trading days x 10 industries:
NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, Other).

To rebuild it from the source:

```bash
python scripts/download_data.py
```

This saves the raw zip under `data/raw/` (not committed) and rewrites the processed CSV.

In code:

```python
from src.data import load_returns, walk_forward_splits

returns = load_returns()
for test_year, train, test in walk_forward_splits(returns):  # train = 5 years before test_year
    ...
```

## Structure

```
data/
  raw/                 downloaded zip (gitignored)
  processed/           industry10_daily_2000_2024.csv (committed)
scripts/
  download_data.py     download + parse the Kenneth French CSV
src/
  data.py              load_returns(), walk_forward_splits()
  models/
    base.py            RiskModel interface: fit(train_returns), sample(n) -> (n, 10)
    ddpm.py            diffusion model (stub)
    vae.py             VAE (stub)
    historical.py      historical simulation (stub)
    gaussian_lw.py     normal model with Ledoit-Wolf covariance (stub)
  eval/                VaR/ES, Kupiec, Christoffersen, ES backtest (TODO)
tests/
  test_data.py
```

Every model subclasses `RiskModel`, so evaluation code can treat all of them the same way.

## Ownership

| Member | Responsibility |
| --- | --- |
| Karla Lučić | DDPM |
| Borna Matulić | VAE |
| Emir Jusić | Data, walk-forward, baselines |
| Domagoj Žokalj | VaR/ES, backtesting |
| Andrija Petrušić | Realism metrics, visualizations |
