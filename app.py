import os
import typer
import json
from groq import Groq
from config import load_config
import yfinance as yf
from dotenv import load_dotenv
import numpy as np
import pandas as pd
load_dotenv()

app = typer.Typer()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


current_market = "IN"


EXCHANGE_SUFFIXES = {"NS", "BO"}


def resolve_ticker(ticker: str, market: str = "IN") -> str:
    """
    Resolve a bare ticker to the correct yfinance symbol.
    Known exchange suffixes (.NS, .BO) are kept as-is. In US mode, any other
    "." is a share class and becomes "-" (BRK.B -> BRK-B, as yfinance expects).
    Otherwise, append .NS for India, or leave bare for US.
    """
    ticker = ticker.upper()
    if "." in ticker:
        if ticker.rsplit(".", 1)[1] in EXCHANGE_SUFFIXES:
            return ticker
        return ticker.replace(".", "-") if market.upper() == "US" else ticker
    if market.upper() == "US":
        return ticker
    return f"{ticker}.NS"


def compute_live_var(ticker, market="IN", window=252, target_alpha=0.01, gamma=0.01, alpha_bounds=(0.001, 0.03)):
    """Live ACI VaR: replays the ACI update rule to get today's adapted VaR"""
    display_ticker = ticker.upper()
    resolved = resolve_ticker(ticker, market)
    t = yf.Ticker(resolved)
    data = t.history(period="2y")
    if data.empty or len(data) < window + 30:
        return {"error": "Not enough data"}

    returns = data["Close"].pct_change().dropna()
    losses = -returns

    alpha_t = target_alpha
    for i in range(window, len(losses)):
        window_losses = losses.iloc[i - window:i]
        aci_var = np.quantile(window_losses, 1 - alpha_t)
        err_t = 1 if losses.iloc[i] > aci_var else 0
        alpha_t = np.clip(alpha_t + gamma * (target_alpha - err_t), *alpha_bounds)

    final_window = losses.iloc[-window:]
    current_var = np.quantile(final_window, 1 - alpha_t)
    level = round((1 - alpha_t) * 100, 2)

    return {
        "ticker": display_ticker,
        "var_1day_pct": round(current_var * 100, 2),
        "quantile_level_pct": level,
        "adapted_alpha": round(alpha_t, 4),
        "target_alpha": target_alpha,
        "interpretation": f"1-day loss threshold at the ACI-adapted {level}% quantile (long run coverage target: {(1 - target_alpha) * 100:g}%)",
    }


MARKET_LABELS = {"IN": "India (NSE)", "US": "United States"}
CURRENCY = {"IN": "INR", "US": "USD"}

# Mocked demo portfolios; Kite Connect integration deferred.
PORTFOLIOS = {
    "IN": [{"ticker": "TCS", "qty": 10}, {"ticker": "RELIANCE", "qty": 5}, {"ticker": "INFY", "qty": 15}],
    "US": [{"ticker": "AAPL", "qty": 10}, {"ticker": "MSFT", "qty": 5}, {"ticker": "NVDA", "qty": 15}],
}


def normalize_market(market: str) -> str:
    m = market.upper()
    if m not in MARKET_LABELS:
        raise typer.BadParameter("market must be IN or US")
    return m

STALE_MAX_ROWS = 3 # max trading days a holding;s last valid price may lag the porfolio's newest row


def get_exposure(market: str = "IN") -> dict:
    """Weights, values and correlation for the mocked portfolio of a market."""
    holdings_list = PORTFOLIOS[market]
    symbols = [resolve_ticker(h["ticker"], market) for h in holdings_list]
    data = yf.download(symbols, period="6mo")["Close"]
    missing = [s for s in symbols if s not in data.columns or data[s].isna().all()]
    if missing:
        return{"error": f"No price data for: {', '.join(missing)}"}
    latest_prices, stale, too_old = {}, [], []
    for h, s in zip(holdings_list, symbols):
        last_valid = data[s].last_valid_index()
        lag = len(data) - 1 - data.index.get_loc(last_valid)
        if lag > STALE_MAX_ROWS:
            too_old.append(h["ticker"])
        elif lag > 0:
            stale.append({"ticker": h["ticker"], "as_of": str(last_valid.date())})
        latest_prices[s] = float(data[s].loc[last_valid])
    if too_old:
        return{"error": f"Price data older than {STALE_MAX_ROWS} trading days for: {', '.join(too_old)}"}

    values = {h["ticker"]: latest_prices[s] * h["qty"] for h, s in zip(holdings_list, symbols)}
    total_value = sum(values.values())

    return {
        "market": market,
        "currency": CURRENCY[market],
        "stale": stale,
        "holdings_value": {t: round(v, 2) for t, v in values.items()},
        "weights_pct": {t: round(v / total_value * 100, 1) for t, v in values.items()},
        "total_value": round(total_value, 2),
        "correlation_matrix": data.pct_change().dropna().corr().round(2).to_dict(),
    }

def fetch_quote(ticker: str, market: str = "IN") -> dict:
    """Latest price and day change; shared by the quote CLI and the get_quote tool."""
    data = yf.Ticker(resolve_ticker(ticker, market)).history(period="5d")
    if data.empty or len(data) < 2:
        return {"error": "Not enough data found"}
    latest = data["Close"].iloc[-1]
    previous = data["Close"].iloc[-2]
    return {
        "ticker": ticker.upper(),
        "price": round(latest, 2),
        "change": round(latest - previous, 2),
        "pct_change": round((latest - previous) / previous * 100, 2),
    }


def fetch_fundamentals(ticker: str, market: str = "IN") -> dict:
    """P/E, market cap, ROE (percent) and D/E (ratio); shared by the fundamentals CLI and the tool."""
    info = yf.Ticker(resolve_ticker(ticker, market)).info or {}
    roe = info.get("returnOnEquity")
    de = info.get("debtToEquity")
    return {
        "ticker": ticker.upper(),
        "pe": info.get("trailingPE"),
        "market_cap": info.get("marketCap"),
        "roe_pct": round(roe * 100, 2) if roe is not None else None,
        "de_ratio": round(de / 100, 2) if de is not None else None,
    }


# [CHANGE 1] Single shared system prompt + tool-round cap
SYSTEM_PROMPT = (
    "You are a financial data assistant. Only state numbers that come from tool results. "
    "Never estimate, infer, or state a figure (volatility, beta, historical loss, ratios, etc) "
    "that wasn't explicitly returned by a tool. If asked for something not covered by your tools, "
    "say so clearly. Do not introduce comparative statistics, industry benchmarks, or 'typical range' "
    "claims unless a tool explicitly returned them - even when reasoning about a number a tool did return. "
    "If a tool result lists stale holdings, say which ones and their as_of dates."
    )
MAX_ROUNDS = 6


def system_prompt_for(market: str) -> str:
    return SYSTEM_PROMPT + f" The market for this session is {MARKET_LABELS[market]}; tickers are resolved for that market."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get the current stock price and day change for a ticker",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. TCS or AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Get P/E, market cap, ROE (as a percent), and D/E for a ticker",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. TCS or AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_var",
            "description": "Get the 1-day Value at Risk (VaR) estimate at an ACI-adapted quantile level (long-run coverage target 99%) for a ticker",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. TCS or AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exposure",
            "description": "Get weight breakdown and correlation matrix for the mocked demo portfolio in the current market",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def run_tool(name, args):
    """Executes the real function based on what the model requested.
    Uses the module-level current_market, set by chat() at session start
    (defaults to India for ask(), which has no market prompt)."""
    if name == "get_quote":
        return fetch_quote(args["ticker"], current_market)
    elif name == "get_fundamentals":
        return fetch_fundamentals(args["ticker"], current_market)
    elif name == "get_var":
        return compute_live_var(args["ticker"], current_market)
    elif name == "get_exposure":
        return get_exposure(current_market)
    return {"error": "Unknown tool"}


def safe_run_tool(name, args_json):
    """Wraps run_tool so bad JSON or a yfinance failure returns an error dict
    to the model instead of crashing the session."""
    try:
        return run_tool(name, json.loads(args_json))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.command()
def quote(ticker: str, market: str = typer.Option("IN", help="IN for India (NSE) or US for United States")):
    """Get quote for a ticker"""
    result = fetch_quote(ticker, (market))
    if "error" in result:
        print(f"No data found for {ticker.upper()}")
        return
    print(f"{result['ticker']}: {result['price']:.2f}  ({result['change']:+.2f}, {result['pct_change']:+.2f}%)")

@app.command()
def chat():
    """Start an interactive research session with the AI analyst"""
    global current_market

    print("Which market are you researching?")
    print("  [1] India (NSE)")
    print("  [2] United States")
    choice = input("> ").strip()
    current_market = "US" if choice == "2" else "IN"
    print(f"Market set to {'United States' if current_market == 'US' else 'India (NSE)'}.\n")

    messages = [
        {"role": "system", "content": system_prompt_for(current_market)}
    ]
    print("Chat session started. Type 'exit' to quit.\n")

    while True:
        user_input = input("> ")
        if user_input.strip().lower() in ("exit", "quit"):
            print("Session ended.")
            break

        messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_ROUNDS):  # [CHANGE 3] was: while True
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            message = response.choices[0].message

            if not message.tool_calls:
                print(message.content + "\n")
                messages.append({"role": "assistant", "content": message.content})
                break

            messages.append(message)
            for tool_call in message.tool_calls:
                # [CHANGE 2] was: json.loads + run_tool
                result = safe_run_tool(tool_call.function.name, tool_call.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:  # [CHANGE 3] runs only if the loop hit MAX_ROUNDS without break
            print("Stopped: too many tool rounds without a final answer.\n")


@app.command()
def ask(query: str, market: str = typer.Option("IN", help="IN for India (NSE) or US for United States")):
    """Ask the AI analyst a one-shot question."""
    global current_market
    current_market = normalize_market(market)
    messages = [
        {"role": "system", "content": system_prompt_for(current_market)},
        {"role": "user", "content": query}
    ]

    for _ in range(MAX_ROUNDS):  # [CHANGE 3] was: while True
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        message = response.choices[0].message

        if not message.tool_calls:
            print(message.content)
            return

        messages.append(message)

        for tool_call in message.tool_calls:
            # [CHANGE 2] was: json.loads + run_tool
            result = safe_run_tool(tool_call.function.name, tool_call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    print("Stopped: too many tool rounds without a final answer.")  # [CHANGE 3]


@app.command()
def exposure(market: str = typer.Option("IN", help="IN for India (NSE) or US for United States")):
    """Show exposure and concentration risk for the mocked demo portfolio"""
    result = get_exposure(normalize_market(market))
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    cur = result["currency"]
    print(f"Portfolio Exposure ({MARKET_LABELS[result['market']]}, mocked):")
    for ticker, value in result["holdings_value"].items():
        print(f"  {ticker}: {cur} {value:,.2f}  ({result['weights_pct'][ticker]:.1f}% of portfolio)")
    for s in result["stale"]:
        print(f" Warning: {s['ticker']} priced as of {s['as_of']} (latest price missing)")
    print(f"\nTotal Portfolio Value: {cur} {result['total_value']:,.2f}")
    print("\nCorrelation Matrix:")
    print(pd.DataFrame(result["correlation_matrix"]))


@app.command()
def fundamentals(ticker: str, market: str = typer.Option("IN", help="IN for India (NSE) or US for United States")):
    """Get fundamentals for a ticker"""
    result = fetch_fundamentals(ticker, market)

    if result["pe"] is None:
        print(f"No fundamentals data found for {result['ticker']}")
        return

    pe = result["pe"]
    market_cap = result["market_cap"]
    roe = result["roe_pct"]
    de = result["de_ratio"]

    print(f"{result['ticker']} Fundamentals:")
    print(f"  P/E Ratio: {pe:.2f}")
    print(f"  Market Cap: {market_cap:,}" if market_cap is not None else "  Market Cap: N/A")
    print(f"  ROE: {roe:.2f}%" if roe is not None else "  ROE: N/A")
    print(f"  D/E: {de:.2f}" if de is not None else "  D/E: N/A")


@app.command()
def var(ticker: str, market: str = typer.Option("IN", help="IN for India (NSE) or US for United States")):
    """Get adaptive VaR estimate for a ticker"""
    result = compute_live_var(ticker, market)
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    print(f"{result['ticker']} — 1-Day VaR ({result['quantile_level_pct']}% adapted quantile): {result['var_1day_pct']}%")
    print(f"Adapted alpha: {result['adapted_alpha']} (target: {result['target_alpha']})")
    print(result["interpretation"])


if __name__ == "__main__":
    load_config()
    app()