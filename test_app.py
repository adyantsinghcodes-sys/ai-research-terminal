import os

# app.py builds a Groq client at import time; make sure that never fails in tests.
os.environ.setdefault("GROQ_API_KEY", "test-key")

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from typer.testing import CliRunner

import app


class FakeTicker:
    """Stand-in for yfinance.Ticker that returns canned history and never touches the network."""

    def __init__(self, symbol, history_df=None, info=None):
        self.symbol = symbol
        self._history = history_df if history_df is not None else pd.DataFrame()
        self.info = info or {}

    def history(self, period=None):
        return self._history


def make_history(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.015, rows)
    close = 100 * np.cumprod(1 + returns)
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame({"Close": close}, index=index)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail loudly if any test reaches real yfinance."""
    def _no_network(*args, **kwargs):
        raise AssertionError("yfinance was called without a mock")

    monkeypatch.setattr(app.yf, "Ticker", _no_network)
    monkeypatch.setattr(app.yf, "download", _no_network)
    monkeypatch.setattr(app, "groq_client", SimpleNamespace(chat=None))
    monkeypatch.setattr(app, "current_market", "IN")


# ---------- safe_run_tool error paths ----------

def test_safe_run_tool_malformed_json():
    result = app.safe_run_tool("get_quote", "{not valid json")
    assert set(result) == {"error"}
    assert result["error"].startswith("JSONDecodeError:")


def test_safe_run_tool_missing_argument():
    result = app.safe_run_tool("get_quote", "{}")
    assert set(result) == {"error"}
    assert result["error"].startswith("KeyError:")
    assert "ticker" in result["error"]


def test_safe_run_tool_yfinance_raises(monkeypatch):
    def boom(symbol):
        raise ConnectionError("network down")

    monkeypatch.setattr(app.yf, "Ticker", boom)
    result = app.safe_run_tool("get_quote", '{"ticker": "TCS"}')
    assert result == {"error": "ConnectionError: network down"}


def test_safe_run_tool_unknown_tool():
    assert app.safe_run_tool("does_not_exist", "{}") == {"error": "Unknown tool"}


# ---------- resolve_ticker ----------

@pytest.mark.parametrize("ticker, market, expected", [
    ("tcs", "IN", "TCS.NS"),
    ("RELIANCE", "in", "RELIANCE.NS"),
    ("aapl", "US", "AAPL"),
    ("MSFT", "us", "MSFT"),
    ("tcs.bo", "IN", "TCS.BO"),
    ("INFY.NS", "US", "INFY.NS"),
    ("brk.b", "US", "BRK-B"),
])
def test_resolve_ticker(ticker, market, expected):
    assert app.resolve_ticker(ticker, market) == expected


def test_resolve_ticker_defaults_to_india():
    assert app.resolve_ticker("infy") == "INFY.NS"


# ---------- compute_live_var ----------

def test_compute_live_var_key_names(monkeypatch):
    requested = []

    def fake_ticker(symbol):
        requested.append(symbol)
        return FakeTicker(symbol, make_history())

    monkeypatch.setattr(app.yf, "Ticker", fake_ticker)
    result = app.compute_live_var("tcs", "IN")

    assert requested == ["TCS.NS"]
    assert list(result.keys()) == [
        "ticker",
        "var_1day_pct",
        "quantile_level_pct",
        "adapted_alpha",
        "target_alpha",
        "interpretation",
    ]
    assert result["ticker"] == "TCS"
    assert result["target_alpha"] == 0.01
    assert "99%" in result["interpretation"]


def test_compute_live_var_not_enough_data(monkeypatch):
    monkeypatch.setattr(app.yf, "Ticker", lambda s: FakeTicker(s, make_history(rows=100)))
    assert app.compute_live_var("TCS") == {"error": "Not enough data"}


def test_get_var_via_safe_run_tool_uses_current_market(monkeypatch):
    requested = []

    def fake_ticker(symbol):
        requested.append(symbol)
        return FakeTicker(symbol, make_history())

    monkeypatch.setattr(app.yf, "Ticker", fake_ticker)
    monkeypatch.setattr(app, "current_market", "US")
    result = app.safe_run_tool("get_var", '{"ticker": "aapl"}')

    assert requested == ["AAPL"]
    assert "error" not in result


# ---------- get_fundamentals ----------

def test_get_fundamentals_converts_yfinance_units(monkeypatch):
    info = {"returnOnEquity": 0.085, "debtToEquity": 46.28, "trailingPE": 25.0, "marketCap": 1_000_000}
    monkeypatch.setattr(app.yf, "Ticker", lambda s: FakeTicker(s, info=info))
    result = app.run_tool("get_fundamentals", {"ticker": "tcs"})

    assert result["roe_pct"] == 8.5
    assert result["de_ratio"] == 0.46


def test_get_fundamentals_missing_fields_are_none_and_zero_is_kept(monkeypatch):
    monkeypatch.setattr(app.yf, "Ticker", lambda s: FakeTicker(s, info={}))
    missing = app.run_tool("get_fundamentals", {"ticker": "tcs"})
    assert missing["roe_pct"] is None
    assert missing["de_ratio"] is None

    zero_info = {"returnOnEquity": 0, "debtToEquity": 0}
    monkeypatch.setattr(app.yf, "Ticker", lambda s: FakeTicker(s, info=zero_info))
    zero = app.run_tool("get_fundamentals", {"ticker": "tcs"})
    assert zero["roe_pct"] is not None and zero["roe_pct"] == 0
    assert zero["de_ratio"] is not None and zero["de_ratio"] == 0


# ---------- var CLI ----------

def test_var_cli_reads_compute_live_var_keys(monkeypatch):
    from typer.testing import CliRunner

    # Same keys as test_compute_live_var_key_names; renaming one there without
    # updating the CLI should surface here as a KeyError.
    fake_result = {
        "ticker": "TCS",
        "var_1day_pct": 2.31,
        "quantile_level_pct": 98.8,
        "adapted_alpha": 0.012,
        "target_alpha": 0.01,
        "interpretation": "Test interpretation.",
    }
    monkeypatch.setattr(app, "compute_live_var", lambda ticker, market: fake_result)

    result = CliRunner().invoke(app.app, ["var", "tcs"])

    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0
    assert "TCS" in result.output
    assert "2.31%" in result.output
    assert "Test interpretation." in result.output


# ---------- get_exposure ----------

def make_close_frame(symbols, rows=130, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    return pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0, 0.01, rows)) for s in symbols}, index=index)


def fake_download_recorder(requested):
    def fake_download(tickers, period=None):
        requested.append(list(tickers))
        return {"Close": make_close_frame(tickers)}
    return fake_download


@pytest.mark.parametrize("market, expected_symbols, currency", [
    ("IN", ["TCS.NS", "RELIANCE.NS", "INFY.NS"], "INR"),
    ("US", ["AAPL", "MSFT", "NVDA"], "USD"),
])
def test_get_exposure_tool_follows_current_market(monkeypatch, market, expected_symbols, currency):
    requested = []
    monkeypatch.setattr(app.yf, "download", fake_download_recorder(requested))
    monkeypatch.setattr(app, "current_market", market)

    result = app.safe_run_tool("get_exposure", "{}")

    assert "error" not in result, result
    assert requested == [expected_symbols]
    assert result["currency"] == currency
    assert abs(sum(result["weights_pct"].values()) - 100) < 0.5


def test_exposure_cli_us_market(monkeypatch):
    requested = []
    monkeypatch.setattr(app.yf, "download", fake_download_recorder(requested))

    result = CliRunner().invoke(app.app, ["exposure", "--market", "US"])

    assert result.exit_code == 0, repr(result.exception)
    assert requested == [["AAPL", "MSFT", "NVDA"]]
    assert "USD" in result.output and "INR" not in result.output


# ---------- ask ----------

def test_ask_market_option_sets_market_and_tells_model(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(app, "groq_client", fake_client)

    result = CliRunner().invoke(app.app, ["ask", "AAPL price", "--market", "US"])

    assert result.exit_code == 0, repr(result.exception)
    assert app.current_market == "US"
    assert "United States" in captured["messages"][0]["content"]

def test_get_exposure_reports_failed_download(monkeypatch):
    def fake_download(tickers, period=None):
        frame = make_close_frame(tickers)
        frame["MSFT"] = np.nan
        return {"Close": frame}

    monkeypatch.setattr(app.yf, "download", fake_download)
    assert app.get_exposure("US") == {"error": "No price data for: MSFT"}