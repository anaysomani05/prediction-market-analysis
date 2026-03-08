"""Fetch sample data from Kalshi and Polymarket public APIs for backtesting.

Pulls resolved markets and their trades from both platforms using free,
unauthenticated endpoints.  Normalizes Polymarket data into the same
column format as Kalshi so the backtesting engine works on both without
changes.

Usage:
    uv run python scripts/fetch_sample_data.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pandas as pd
from tqdm import tqdm

from src.indexers.kalshi.client import KalshiClient

SAMPLE_DIR = Path("data/sample")

# ── Kalshi ──────────────────────────────────────────────────────────────

KALSHI_MARKETS_DIR = SAMPLE_DIR / "markets"
KALSHI_TRADES_DIR = SAMPLE_DIR / "trades"

KALSHI_TARGET = 1000
KALSHI_MIN_VOLUME = 50


def _kalshi_recent_windows(n_windows: int = 6) -> list[dict]:
    """Build time windows dynamically from current time, going backwards."""
    now = int(time.time())
    window_size = 90 * 24 * 3600  # ~90 days per window
    windows = []
    for i in range(n_windows):
        end = now - i * window_size
        start = end - window_size
        label = (
            datetime.fromtimestamp(start, tz=timezone.utc).strftime("%b %Y")
            + " – "
            + datetime.fromtimestamp(end, tz=timezone.utc).strftime("%b %Y")
        )
        windows.append({"label": label, "min_close_ts": start, "max_close_ts": end})
    return list(reversed(windows))


def fetch_kalshi():
    """Fetch resolved Kalshi markets and trades."""
    KALSHI_MARKETS_DIR.mkdir(parents=True, exist_ok=True)
    KALSHI_TRADES_DIR.mkdir(parents=True, exist_ok=True)

    client = KalshiClient()
    windows = _kalshi_recent_windows()
    per_window = KALSHI_TARGET // len(windows)

    print(f"[Kalshi] Fetching up to {KALSHI_TARGET} resolved markets "
          f"({per_window}/window, min volume {KALSHI_MIN_VOLUME})...")

    all_markets: list[dict] = []

    for window in windows:
        collected: list[dict] = []
        cursor = None
        print(f"\n  Window: {window['label']}")

        while len(collected) < per_window:
            params: dict = {
                "limit": 200,
                "min_close_ts": window["min_close_ts"],
                "max_close_ts": window["max_close_ts"],
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = client._get("/markets", params=params)
            except Exception as e:
                print(f"  API error: {e}")
                break

            for m in data.get("markets", []):
                if (m.get("result") in ("yes", "no")
                        and m.get("volume", 0) >= KALSHI_MIN_VOLUME):
                    collected.append(m)
                    if len(collected) >= per_window:
                        break

            cursor = data.get("cursor")
            if not cursor:
                break

        print(f"  Found {len(collected)} markets")
        all_markets.extend(collected)

    if not all_markets:
        print("[Kalshi] No markets found.")
        client.close()
        return

    # Save markets
    markets_df = pd.DataFrame(all_markets)
    markets_df.to_parquet(KALSHI_MARKETS_DIR / "markets.parquet")
    print(f"\n[Kalshi] Saved {len(markets_df)} markets")

    # Fetch trades
    print(f"[Kalshi] Fetching trades for {len(all_markets)} markets...")
    all_trades: list[dict] = []
    fetched_at = datetime.now(tz=timezone.utc)

    for market in tqdm(all_markets, desc="  Trades"):
        ticker = market["ticker"]
        try:
            trades = client.get_market_trades(ticker, verbose=False)
            for t in trades:
                d = asdict(t)
                d["_fetched_at"] = fetched_at
                all_trades.append(d)
        except Exception as e:
            tqdm.write(f"  Error fetching {ticker}: {e}")

    client.close()

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_parquet(KALSHI_TRADES_DIR / "trades.parquet")
        print(f"[Kalshi] Saved {len(trades_df)} trades")
    else:
        print("[Kalshi] No trades fetched.")

    return len(all_markets), len(all_trades)


# ── Polymarket ──────────────────────────────────────────────────────────

POLY_MARKETS_DIR = SAMPLE_DIR / "polymarket_markets"
POLY_TRADES_DIR = SAMPLE_DIR / "polymarket_trades"

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

POLY_TARGET = 500
POLY_MIN_VOLUME = 1000  # USD volume


def _api_get(base_url: str, path: str, params: dict | None = None) -> list | dict:
    """GET from a Polymarket API with basic retry."""
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(3):
            try:
                r = client.get(f"{base_url}{path}", params=params)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPStatusError, httpx.ConnectError) as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return []


def _gamma_get(path: str, params: dict | None = None) -> list | dict:
    return _api_get(GAMMA_API, path, params)


def _data_api_get_trades(condition_id: str, limit: int = 500) -> list[dict]:
    """Fetch trades for a market from the Polymarket Data API."""
    try:
        data = _api_get(DATA_API, "/trades", params={
            "conditionId": condition_id,
            "limit": limit,
        })
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _poly_resolve_outcome(outcome_prices_str: str) -> str | None:
    """Determine if a Polymarket market resolved YES or NO."""
    try:
        prices = json.loads(outcome_prices_str)
        if not prices or len(prices) < 2:
            return None
        # prices[0] = YES payout, prices[1] = NO payout
        yes_price = float(prices[0])
        no_price = float(prices[1])
        if yes_price >= 0.99:
            return "yes"
        if no_price >= 0.99:
            return "no"
        return None  # not cleanly resolved
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def fetch_polymarket():
    """Fetch resolved Polymarket markets and trades."""
    POLY_MARKETS_DIR.mkdir(parents=True, exist_ok=True)
    POLY_TRADES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Polymarket] Fetching up to {POLY_TARGET} resolved markets "
          f"(min volume ${POLY_MIN_VOLUME})...")

    all_markets: list[dict] = []
    offset = 0
    batch_size = 100

    while len(all_markets) < POLY_TARGET:
        try:
            data = _gamma_get("/markets", params={
                "limit": batch_size,
                "offset": offset,
                "closed": "true",
                "order": "volume",
                "ascending": "false",
            })
        except Exception as e:
            print(f"  API error: {e}")
            break

        if not data:
            break

        markets = data if isinstance(data, list) else data.get("markets", data)
        if not markets:
            break

        for m in markets:
            result = _poly_resolve_outcome(
                m.get("outcomePrices", m.get("outcome_prices", "[]"))
            )
            if result is None:
                continue
            vol = float(m.get("volume", 0) or 0)
            if vol < POLY_MIN_VOLUME:
                continue

            token_ids_raw = m.get("clobTokenIds", m.get("clob_token_ids", "[]"))
            try:
                token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
            except json.JSONDecodeError:
                continue
            if not token_ids or len(token_ids) < 2:
                continue

            all_markets.append({
                "raw": m,
                "result": result,
                "yes_token": token_ids[0],
                "no_token": token_ids[1],
                "condition_id": m.get("conditionId", m.get("condition_id", "")),
                "question": m.get("question", ""),
                "volume": vol,
                "end_date": m.get("endDate", m.get("end_date", "")),
            })

            if len(all_markets) >= POLY_TARGET:
                break

        print(f"  Scanned {offset + len(markets)} markets, "
              f"found {len(all_markets)} qualifying...")

        offset += len(markets)
        if len(markets) < batch_size:
            break

    if not all_markets:
        print("[Polymarket] No resolved markets found.")
        return

    print(f"[Polymarket] Found {len(all_markets)} resolved markets. "
          "Fetching trades...")

    # Normalize markets to Kalshi-compatible parquet format
    market_rows: list[dict] = []
    trade_rows: list[dict] = []
    fetched_at = datetime.now(tz=timezone.utc)
    skipped = 0

    for entry in tqdm(all_markets, desc="  Trades"):
        cid = entry["condition_id"]
        ticker = f"POLY-{cid[:14]}"  # short unique ticker
        result = entry["result"]

        # Fetch trades via Data API (uses conditionId)
        raw_trades = _data_api_get_trades(cid)
        if not raw_trades:
            skipped += 1
            continue

        # Figure out which outcomeIndex is YES (index 0) vs NO (index 1)
        for t in raw_trades:
            price_raw = t.get("price")
            size_raw = t.get("size")
            side = t.get("side", "").upper()
            outcome_idx = int(t.get("outcomeIndex", 0) or 0)
            ts = t.get("timestamp")

            if price_raw is None or size_raw is None:
                continue

            price = float(price_raw)
            size = float(size_raw)
            if price <= 0 or price >= 1 or size <= 0:
                continue

            # outcomeIndex 0 = YES, 1 = NO
            # price is the price of the outcome being traded
            if outcome_idx == 0:
                # Trading YES outcome
                yes_price_cents = max(1, min(99, round(price * 100)))
                taker_side = "yes" if side == "BUY" else "no"
            else:
                # Trading NO outcome — invert for yes_price
                yes_price_cents = max(1, min(99, 100 - round(price * 100)))
                taker_side = "no" if side == "BUY" else "yes"

            no_price_cents = 100 - yes_price_cents

            # Parse timestamp
            if isinstance(ts, (int, float)):
                created_time = datetime.fromtimestamp(ts, tz=timezone.utc)
            elif isinstance(ts, str):
                try:
                    created_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    created_time = fetched_at
            else:
                created_time = fetched_at

            trade_rows.append({
                "trade_id": t.get("transactionHash", f"{ticker}-{len(trade_rows)}"),
                "ticker": ticker,
                "count": max(1, int(size)),
                "yes_price": yes_price_cents,
                "no_price": no_price_cents,
                "taker_side": taker_side,
                "created_time": created_time,
                "_fetched_at": fetched_at,
            })

        market_rows.append({
            "ticker": ticker,
            "event_ticker": f"POLY-{entry['question'][:30]}",
            "status": "finalized",
            "result": result,
            "volume": int(entry["volume"]),
            "close_time": entry["end_date"],
            "title": entry["question"],
        })

        # Small delay to respect rate limits
        time.sleep(0.15)

    if skipped:
        print(f"  Skipped {skipped} markets (no trades from CLOB API)")

    if not market_rows:
        print("[Polymarket] No markets with trades.")
        return

    # Save
    markets_df = pd.DataFrame(market_rows)
    markets_df.to_parquet(POLY_MARKETS_DIR / "markets.parquet")
    print(f"[Polymarket] Saved {len(markets_df)} markets")

    if trade_rows:
        trades_df = pd.DataFrame(trade_rows)
        trades_df.to_parquet(POLY_TRADES_DIR / "trades.parquet")
        print(f"[Polymarket] Saved {len(trades_df)} trades")
    else:
        print("[Polymarket] No trades saved.")

    return len(market_rows), len(trade_rows)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Fetching sample data from Kalshi + Polymarket")
    print("=" * 60)

    kalshi_result = fetch_kalshi()
    poly_result = fetch_polymarket()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if kalshi_result:
        print(f"  Kalshi:      {kalshi_result[0]} markets, {kalshi_result[1]} trades")
    if poly_result:
        print(f"  Polymarket:  {poly_result[0]} markets, {poly_result[1]} trades")
    print(f"  Data dir:    {SAMPLE_DIR.resolve()}")


if __name__ == "__main__":
    main()
