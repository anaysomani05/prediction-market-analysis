"""Calibration curve fitting from historical prediction market data.

Computes the empirical win rate at each price level and returns
calibration offsets (actual_win_rate - implied_probability) that
can be fed directly into the MispricingStrategy.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _calibration_query(
    trades_dir: Path | str,
    markets_dir: Path | str,
    bin_width: int,
    min_samples: int,
    max_close_ts: str | None = None,
) -> pd.DataFrame:
    """Run the calibration SQL and return raw DataFrame.

    Args:
        max_close_ts: If set, only include markets that closed before this
            ISO-8601 timestamp.  Used to enforce train/test separation.
    """
    close_filter = (
        f"AND close_time < '{max_close_ts}'" if max_close_ts else ""
    )
    con = duckdb.connect()
    return con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
              {close_filter}
        ),
        taker_positions AS (
            SELECT
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
            WHERE CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END BETWEEN 1 AND 99
        )
        SELECT
            (FLOOR((price - 1) / {bin_width}) * {bin_width} + {bin_width} / 2 + 1)::INT AS price_bin,
            AVG(won) AS actual_win_rate,
            AVG(price) / 100.0 AS implied_prob,
            COUNT(*) AS n
        FROM taker_positions
        GROUP BY price_bin
        HAVING COUNT(*) >= {min_samples}
        ORDER BY price_bin
        """
    ).df()


def fit_calibration(
    trades_dir: Path | str,
    markets_dir: Path | str,
    bin_width: int = 5,
    min_samples: int = 20,
    max_close_ts: str | None = None,
) -> dict[int, float]:
    """Compute calibration offsets from historical data.

    Groups trades by price bins, calculates actual win rate vs implied
    probability, and returns the offset at each price midpoint.

    Args:
        trades_dir: Path to Parquet trade files.
        markets_dir: Path to Parquet market files.
        bin_width: Width of price bins in cents.
        min_samples: Minimum trades per bin to include.
        max_close_ts: If set, only use markets that closed before this
            ISO-8601 timestamp (for train/test separation).

    Returns:
        Dict mapping price_cents -> edge (actual_win_rate - implied_prob).
    """
    df = _calibration_query(trades_dir, markets_dir, bin_width, min_samples, max_close_ts)

    calibration: dict[int, float] = {}
    for _, row in df.iterrows():
        price = int(row["price_bin"])
        edge = float(row["actual_win_rate"] - row["implied_prob"])
        calibration[price] = round(edge, 4)

    return calibration


def calibration_curve_data(
    trades_dir: Path | str,
    markets_dir: Path | str,
    bin_width: int = 5,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Return calibration curve data for plotting.

    Returns a DataFrame with columns: price_bin, actual_win_rate,
    implied_prob, n — suitable for a predicted-vs-realized plot.
    Uses smaller min_samples than fit_calibration for denser plotting.
    """
    return _calibration_query(trades_dir, markets_dir, bin_width, min_samples)
