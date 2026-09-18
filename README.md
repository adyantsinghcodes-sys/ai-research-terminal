# AI Research Terminal

A command-line financial-research terminal that combines real-time market data with an AI agent layer coupled to adaptive risk modelling. This project was built to make quantitative finance research (ACI for VaR) into something actually usable, not just a backtest script.

## What It Does

Instead of a typical financial dashboard with charts, news, and other data, this simplifies that into a shell-based terminal where the user can run specific commands or ask questions in plain English, and an AI agent decides which tool to call and grounds its answer in real data.

```bash
python app.py quote TCS
python app.py fundamentals TCS
python app.py var TCS
python app.py exposure
python app.py ask "what's the risk on TCS right now"
```

## Features of the Tool

- **`quote`** — live price and day-change for any NSE-listed stock (via yfinance)
- **`fundamentals`** — P/E, Market Cap, ROE, D/E for a ticker
- **`var`** — a live, adaptive 1-day Value at Risk estimate using **Adaptive Conformal Inference (ACI)**, based on the Gibbs & Candès (2021) update rule. Unlike a fixed-alpha historical VaR, this recalibrates its own confidence level every day based on recent breach history — so it widens automatically during volatile regimes (e.g. it responds visibly during the 2020 crash in backtests) instead of lagging behind a static threshold.
- **`exposure`** — portfolio weight breakdown plus a correlation matrix, surfacing concentration risk not obvious from sector labels alone (mock portfolio for now — CLI-only, not yet wired into `ask`)
- **`ask`** — natural-language interface. An LLM (via Groq) is given the above as callable tools, decides which is the most appropriate based on the user's question, and prints an output based strictly on the numbers returned by the tools (the LLM is **constrained** via a system prompt to prevent it from inventing numbers).

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

The `var` command is a live version of a research pipeline built separately (GARCH volatility modelling, Kupiec POF and Christoffersen independence backtesting, and Adaptive Conformal Inference — validated first on synthetic data, then on ^NSEI and ^GSPC historical data). `var` replays the ACI update rule over a trailing window to arrive at today's adapted confidence level, then reports the corresponding VaR.

**Known limitation:** the ACI parameters (`gamma`, `alpha_bounds`) were tuned on index-level data (NIFTY/S&P), not individual stocks — applying them to single tickers is a reasonable extension but hasn't been separately validated.

## Setup

```bash
pip install -r requirements.txt
```

Add a `.env` file:

```
GROQ_API_KEY=your_key_here
KITE_API_KEY=      # optional, not yet used — for future real portfolio integration
```

## Tech Stack

- **Python**, **Typer** (CLI)
- **yfinance** — market data
- **pandas / numpy** — data processing, correlation, VaR calculation
- **Groq** (`openai/gpt-oss-120b`) — tool-calling LLM layer

## Roadmap

- [ ] Session memory in `ask` (multi-turn follow-ups)
- [ ] Real portfolio holdings via Kite Connect (currently uses a mock portfolio)
- [ ] Wire `exposure` into `ask` as a callable tool
- [ ] News feed tool
- [ ] Caching for repeated VaR calls

## Status

Built incrementally as a learning project — Python, pandas, and applied LLM tool-calling — alongside coursework. Exposure/portfolio data currently uses mock holdings; `quote`, `fundamentals`, and `var` are all live. Real brokerage integration is a planned next step.