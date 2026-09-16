# Technology Stock Signal Modeling

A research notebook comparing classical machine learning and an optional LSTM for next-day stock direction. It brings together price ingestion, time-series features, chronological model evaluation, and a simplified payoff simulation.

**Python · pandas · scikit-learn · optional TensorFlow/Keras · yfinance · time-series validation**

## Run the reproducible demo

Use Python 3.11 or 3.12. The default notebook uses synthetic prices and runs the logistic regression and random forest models without downloading data.

```bash
git clone https://github.com/davislaroque/stock-ml-notebook.git
cd stock-ml-notebook
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
jupyter lab tech_signals_ml.ipynb
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Run the notebook cells in order.

For the optional LSTM, install `requirements-lstm.txt` and set `RUN_LSTM = True` in the configuration cell. Set `USE_LIVE_DATA = True` to download the configured tickers. Live mode saves the input CSV under `data/downloads/`; review the tickers and date range before running.

## What the code demonstrates

| Component | Implementation |
| --- | --- |
| Price ingestion | Explicit yfinance adjustment settings and normalized columns |
| Feature engineering | Returns, momentum, volatility, volume z-scores, and an RSI-style feature |
| Temporal validation | Whole-date partitions and purged next-day labels at boundaries |
| Classical models | Logistic regression and random forest with a training-majority baseline |
| Optional LSTM | Training-only scalers and preserved source indices for sequence labels |
| Analysis | Classification metrics, feature importance, and illustrative payoff curves |

The [notebook](tech_signals_ml.ipynb) handles experiments and plots. [stock_features.py](stock_features.py) contains reusable data and sequence helpers. [Tests](tests/test_features.py) check partition boundaries and sequence alignment.

## Data and limitations

`data/sample_prices.csv` contains 480 synthetic rows: three fictional tickers with 160 business dates each, generated with NumPy seed 42. It has no connection to actual securities. Demo metrics establish that the pipeline runs; they do not establish forecasting performance.

The optional download uses [yfinance](https://github.com/ranaroussi/yfinance). Preserve the input snapshot when reporting real-data metrics. The ticker list is selected retrospectively and does not correct for survivorship bias.

The payoff calculation is an educational approximation using fixed delta/premium assumptions, transaction costs, and equal allocation across tickers. It does not price real options or demonstrate a tradable strategy. Signals use closing data; realistic execution would need an entry price available after the signal. The LSTM uses fewer observations because of its lookback window, so compare models on matching rows before ranking them.

Next work: walk-forward evaluation, probability calibration, and consistent test windows across model families.
