from pathlib import Path
import numpy as np
import pandas as pd
from stock_features import (
    FEATURE_COLS,
    engineer_features,
    time_based_split,
    build_sequences,
)


def data():
    return engineer_features(
        pd.read_csv(Path(__file__).parents[1] / "data/sample_prices.csv")
    )


def test_dates_and_forward_labels_do_not_cross_partitions():
    train, val, test = time_based_split(data())
    assert train.target_date.max() < val.date.min()
    assert val.target_date.max() < test.date.min()
    assert set(train.date).isdisjoint(val.date)
    assert set(val.date).isdisjoint(test.date)


def test_sequences_match_label_row_and_reuse_training_scalers():
    train, val, test = time_based_split(data())
    X, y, indices, scalers = build_sequences(train, 5)
    Xtest, ytest, test_indices, _ = build_sequences(test, 5, scalers)
    assert np.array_equal(ytest, test.loc[test_indices, "target"])
    for pos, index in enumerate(test_indices):
        row = test.loc[[index]]
        ticker = row.ticker.iloc[0]
        expected = scalers[ticker].transform(row[FEATURE_COLS])[0]
        assert np.allclose(Xtest[pos, -1], expected)
    for ticker, scaler in scalers.items():
        assert scaler.n_samples_seen_ == len(train.loc[train.ticker == ticker])


def test_feature_rows_exclude_unknown_final_labels():
    frame = data()
    assert frame.target_date.notna().all()
    assert (frame.target_date > frame.date).all()
    assert np.isfinite(frame[FEATURE_COLS]).all().all()
