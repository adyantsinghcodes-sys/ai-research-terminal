# AI Research Terminal

A command-line financial-research terminal that combines market data with an AI agent layer and adaptive risk modelling. Built to make one piece of quantitative finance research (Adaptive Conformal Inference for VaR) usable as a tool, not just a backtest script.

## What It Does

Instead of a dashboard full of charts and news, this is a shell-based terminal: run specific commands, or ask questions in plain English and let an AI agent pick the right tool and ground its answer in the data that tool returns.

Supports India (NSE, default) and US markets.

```bash
python app.py quote TCS
python app.py quote AAPL --market US
python app.py fundamentals TCS
python app.py var TCS
python app.py exposure --market US
python app.py ask "what's the risk on TCS right now"
python app.py ask "how is NVDA doing today" --market US
python app.py chat
```

## Commands

- **`quote`** — latest price and day change for a ticker (via yfinance; prices may be delayed).
- **`fundamentals`** — P/E, market cap, ROE, D/E.
- **`var`** — adaptive 1-day Value at Risk using **Adaptive Conformal Inference (ACI)** (Gibbs & Candès, 2021). See [The VaR Model](#the-var-model).
- **`exposure`** — portfolio weights, total value and a correlation matrix for a **mocked** demo portfolio (separate IN and US portfolios). Returns an error if any holding fails to download, rather than reporting NaN values.
- **`ask`** — one-shot natural-language query. An LLM (Groq, `openai/gpt-oss-120b`) gets the tools above, chooses which to call, and answers using only numbers the tools returned.
- **`chat`** — the same agent as a multi-turn session with conversation memory; asks which market you are researching at the start.

All commands except `chat` take `--market IN` (default) or `--market US`.

Ticker handling: bare tickers get `.NS` in India mode and stay bare in US mode. Explicit `.NS` / `.BO` suffixes are kept. In US mode, share classes like `BRK.B` are converted to yfinance's `BRK-B`.

## Grounding the LLM

The system prompt restricts the model to numbers returned by tools: no estimated figures, no industry benchmarks or "typical ranges" unless a tool returned them. This was added after the model invented sector benchmarks in testing.

## Architecture

```
User query
   │
   ▼
LLM (tool-calling) ──picks a tool──▶ run_tool() ──▶ real function (yfinance / ACI VaR / portfolio math)
   │                                                        │
   └───────────────── real data returned ◀──────────────────┘
   │
   ▼
Final grounded answer
```

## The VaR Model

`var` is a **rolling empirical VaR with ACI-adapted coverage**. There is no GARCH in this command:

1. Take two years of daily returns and treat losses as negative returns.
2. Estimate VaR as the empirical quantile of the trailing 252 days of losses.
3. Replay the ACI update over the history: after each day, the quantile level is nudged up if that day breached the VaR and down if it did not, targeting 99% long-run coverage.
4. Report today's VaR at the adapted quantile level.

Because the quantile level reacts to recent breaches, the estimate tightens faster after a run of large losses than a fixed-quantile historical VaR would. It is still reactive: it adjusts after breaches, it does not anticipate them.

Full research pipeline, backtests and results: [aci-var-research](https://github.com/adyantsinghcodes-sys/aci-var-research).

**Known limitation:** the ACI parameters (`gamma`, `alpha_bounds`) were tuned on index data (NIFTY / S&P 500), not individual stocks. Applying them to single tickers has not been separately validated.

## Setup

```bash
pip install -r requirements.txt
```

Add a `.env` file:

```
GROQ_API_KEY=your_key_here
KITE_API_KEY=      # optional, not yet used — for future real portfolio integration
```

## Running Tests

```bash
pytest
```

The suite runs **fully offline** in a few seconds. yfinance and the Groq client are replaced in every test by an autouse fixture, and any unmocked network call fails the test. No API keys are needed to run the tests.

## Tech Stack

- **Python**, **Typer** (CLI)
- **yfinance** — market data
- **pandas / numpy** — data processing, correlation, VaR
- **Groq** (`openai/gpt-oss-120b`) — tool-calling LLM layer
- **pytest** — offline test suite

## Roadmap

- [ ] Real portfolio holdings via Kite Connect (currently mocked)
- [ ] Validate ACI parameters on individual stocks (realised breach rates vs. 99% target)
- [ ] News feed tool
- [ ] Caching for repeated VaR calls

## Status

Built incrementally as a learning project in Python, pandas and applied LLM tool-calling, alongside coursework. `quote`, `fundamentals` and `var` use live yfinance data; `exposure` uses mocked holdings until brokerage integration is added.