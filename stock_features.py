"""Price data preparation and date-safe partitions for the research notebook."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "ret_ma_5",
    "ret_ma_10",
    "ret_std_10",
    "high_roll_10",
    "low_roll_10",
    "volume_z",
    "rsi",
    "return",
]


def fetch_data(tickers, start, end=None):
    """Download explicit unadjusted OHLC plus adjusted close; normalize yfinance columns."""
    import yfinance as yf

    frames = []
    for ticker in tickers:
        frame = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
        if frame.empty:
            raise ValueError(
                f"No data returned for {ticker}; use the offline fixture or retry later."
            )
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(col).lower() for col in frame.columns]
        frame = frame.rename_axis("date").reset_index()
        frame["ticker"] = ticker
        frames.append(frame)
    if not frames:
        raise ValueError("Provide at least one ticker.")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def engineer_features(df):
    required = {"date", "ticker", "adj close", "close", "high", "low", "volume"}
    if not required <= set(df):
        raise ValueError(f"Missing columns: {sorted(required - set(df))}")
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    if data.duplicated(["ticker", "date"]).any():
        raise ValueError("Expected one price row per ticker and date.")
    feats = []
    for _, g in data.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["return"] = g["adj close"].pct_change(fill_method=None)
        g["next_return"] = g["return"].shift(-1)
        g["target_date"] = g["date"].shift(-1)
        g["ret_ma_5"] = g["return"].rolling(5).mean()
        g["ret_ma_10"] = g["return"].rolling(10).mean()
        g["ret_std_10"] = g["return"].rolling(10).std()
        g["high_roll_10"] = g["high"].rolling(10).max() / g["close"] - 1
        g["low_roll_10"] = g["low"].rolling(10).min() / g["close"] - 1
        g["volume_z"] = (g["volume"] - g["volume"].rolling(20).mean()) / g[
            "volume"
        ].rolling(20).std().replace(0, np.nan)
        up = g["close"].diff().clip(lower=0).rolling(14).mean()
        down = -g["close"].diff().clip(upper=0).rolling(14).mean()
        g["rsi"] = (100 - 100 / (1 + up / (down + 1e-9)) - 50) / 50
        g = g.replace([np.inf, -np.inf], np.nan).dropna(
            subset=FEATURE_COLS + ["next_return", "target_date"]
        )
        g["target"] = (g["next_return"] > 0).astype(int)
        feats.append(g)
    result = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    if result.empty:
        raise ValueError(
            "No usable feature rows; provide at least 21 valid daily observations per ticker."
        )
    return result


def time_based_split(df, test_size=0.15, val_size=0.2):
    """Split whole dates and purge labels crossing the next partition boundary."""
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    dates = sorted(df["date"].unique())
    test_cut = int(len(dates) * (1 - test_size))
    val_cut = int(test_cut * (1 - val_size))
    if not 0 < val_cut < test_cut < len(dates):
        raise ValueError("Insufficient distinct dates for three partitions.")
    val_start, test_start = dates[val_cut], dates[test_cut]
    train = df.loc[(df.date < val_start) & (df.target_date < val_start)].copy()
    val = df.loc[
        (df.date >= val_start) & (df.date < test_start) & (df.target_date < test_start)
    ].copy()
    test = df.loc[df.date >= test_start].copy()
    if any(part.empty for part in (train, val, test)):
        raise ValueError("A partition is empty after purging boundary labels.")
    return tuple(part.sort_values(["date", "ticker"]) for part in (train, val, test))


def build_sequences(df, lookback=10, scalers=None):
    """Fit scalers on training only; reuse them and preserve source-row identity.

    A sequence includes the feature row whose next-day direction is its label.
    Validation/test windows start within their own partitions.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer.")
    fit = scalers is None
    scalers = {} if fit else scalers
    sequences, labels, indices = [], [], []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        if fit:
            scalers[ticker] = StandardScaler().fit(g[FEATURE_COLS])
        if ticker not in scalers:
            raise ValueError(f"No training scaler for {ticker}.")
        values = scalers[ticker].transform(g[FEATURE_COLS])
        for end in range(lookback - 1, len(g)):
            sequences.append(values[end - lookback + 1 : end + 1])
            labels.append(g["target"].iloc[end])
            indices.append(g.index[end])
    if not sequences:
        raise ValueError(
            "No complete sequences; reduce lookback or provide more observations."
        )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        indices,
        scalers,
    )
