import os
import typer
import json
from groq import Groq
from config import load_config
import yfinance as yf
from dotenv import load_dotenv
import numpy as np
load_dotenv()

app = typer.Typer()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def compute_live_var(ticker, window=252, target_alpha=0.01, gamma=0.01, alpha_bounds=(0.001, 0.03)):
    """Live ACI VaR: replays the ACI update rule to get today's adapted VaR"""
    t = yf.Ticker(f"{ticker}.NS")
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

    return {
        "ticker": ticker.upper(),
        "var_99_1day": round(current_var * 100, 2),
        "adapted_alpha": round(alpha_t, 4),
        "target_alpha": target_alpha,
        "interpretation": f"On 99% of days, expect not to lose more than {round(current_var*100, 2)}% in a day"
    }

portfolio = [
    {"ticker": "TCS", "qty": 10},
    {"ticker": "RELIANCE", "qty": 5},
    {"ticker": "INFY", "qty": 15},
]

tools = [
    
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get the current stock price and day change for a ticker",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. TCS"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Get P/E, market cap, ROE, and D/E for a ticker",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. TCS"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "get_var",
        "description": "Get the adaptive 1-day Value at Risk (VaR) estimate at 99% confidence for a ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. TCS"}
            },
            "required": ["ticker"]
        }
    }
    },
   {
    "type": "function",
    "function": {
        "name": "get_exposure",
        "description": "Get portfolio weight breakdown and correlation matrix across holdings",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


]


def run_tool(name, args):
    """Executes the real function based on what the model requested"""
    if name == "get_quote":
        t = yf.Ticker(f"{args['ticker'].upper()}.NS")
        data = t.history(period="5d")
        if data.empty or len(data) < 2:
            return {"error": "Not enough data found"}
        latest = data["Close"].iloc[-1]
        previous = data["Close"].iloc[-2]
        return {
            "ticker": args["ticker"].upper(),
            "price": round(latest, 2),
            "change": round(latest - previous, 2),
            "pct_change": round((latest - previous) / previous * 100, 2)
        }
    elif name == "get_fundamentals":
        t = yf.Ticker(f"{args['ticker'].upper()}.NS")
        info = t.info
        return {
            "ticker": args["ticker"].upper(),
            "pe": info.get("trailingPE"),
            "market_cap": info.get("marketCap"),
            "roe": info.get("returnOnEquity"),
            "de": info.get("debtToEquity")
        }
    elif name == "get_var":
        return compute_live_var(args["ticker"].upper())
        
    
    
    elif name == "get_exposure":
        tickers_ns = [f"{h['ticker']}.NS" for h in portfolio]
        data = yf.download(tickers_ns, period="6mo")["Close"]
        latest_prices = data.iloc[-1]
    
        total_value = 0
        holdings = {}
        for h in portfolio:
            ticker_ns = f"{h['ticker']}.NS"
            price = latest_prices[ticker_ns]
            value = price * h["qty"]
            holdings[h["ticker"]] = round(value, 2)
            total_value += value
    
        weights = {t: round((v / total_value) * 100, 1) for t, v in holdings.items()}
        correlation = data.pct_change().dropna().corr().round(2).to_dict()
    
        return {
            "holdings_value": holdings,
            "weights_pct": weights,
            "total_value": round(total_value, 2),
            "correlation_matrix": correlation
        }
    return {"error": "Unknown tool"}
    
 



@app.command()
def quote(ticker: str):
    """Get quote for a ticker"""
    ticker = ticker.upper()
    t = yf.Ticker(f"{ticker}.NS")
    data = t.history(period="5d")

    if data.empty or len(data) < 2:
        print(f"No data found for {ticker}")
        return

    latest = data["Close"].iloc[-1]
    previous = data["Close"].iloc[-2]
    change = latest - previous
    pct_change = (change / previous) * 100

    print(f"{ticker}: {latest:.2f}  ({change:+.2f}, {pct_change:+.2f}%)")

@app.command()
def chat():
    """Start an interactive research session with the AI analyst"""
    messages = [
    {"role": "system", "content": "You are a financial data assistant. Only state numbers that come from tool results. Never estimate, infer, or state a figure that wasn't explicitly returned by a tool. If asked for something not covered by your tools, say so clearly. Do not introduce comparative statistics, industry benchmarks, or 'typical range' claims unless a tool explicitly returned "
    "them — even when reasoning about a number a tool did return."}]
    print("Chat session started. Type 'exit' to quit.\n")

    while True:
        user_input = input("> ")
        if user_input.strip().lower() in ("exit", "quit"):
            print("Session ended.")
            break

        messages.append({"role": "user", "content": user_input})

        while True:
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
                args = json.loads(tool_call.function.arguments)
                result = run_tool(tool_call.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })


@app.command()
def ask(query: str):
    """Ask the AI analyst a question"""
    messages = [
        {"role": "system", "content": "You are a financial data assistant. Only state numbers that come from tool results. Never estimate, infer, or state a figure (volatility, beta, historical loss, ratios, etc) that wasn't explicitly returned by a tool. If asked for something not covered by your tools, say so clearly."},
    {"role": "user", "content": query}
    ]

    while True:
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
            args = json.loads(tool_call.function.arguments)
            result = run_tool(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })


@app.command()
def exposure():
    """Show portfolio exposure and concentration risk"""
    tickers_ns =[f"{h['ticker']}.NS" for h in portfolio]
    data = yf.download(tickers_ns, period="6mo")["Close"]
    latest_prices = data.iloc[-1]

    total_value = 0
    holdings_value={}
    for h in portfolio:
        ticker_ns = f"{h['ticker']}.NS"
        price = latest_prices[ticker_ns]
        value = price*h["qty"]
        holdings_value[h["ticker"]] = value
        total_value += value

    print("Portfolio Exposure:")
    for ticker, value in holdings_value.items():
        weight = (value / total_value) * 100
        print(f"  {ticker}: ₹{value:,.2f}  ({weight:.1f}% of portfolio)")

    print(f"\nTotal Portfolio Value: ₹{total_value:,.2f}")

    returns = data.pct_change().dropna()
    correlation = returns.corr()
    print("\nCorrelation Matrix:")
    print(correlation.round(2))




@app.command()
def fundamentals(ticker: str):
    """Get fundamentals for a ticker"""
    ticker = ticker.upper()
    t = yf.Ticker(f"{ticker}.NS")
    info = t.info

    if not info or info.get("trailingPE") is None:
        print(f"No fundamentals data found for {ticker}")
        return

    pe = info.get("trailingPE")
    market_cap = info.get("marketCap")
    roe = info.get("returnOnEquity")
    de = info.get("debtToEquity")

    print(f"{ticker} Fundamentals:")
    print(f"  P/E Ratio: {pe:.2f}" if pe else "  P/E Ratio: N/A")
    print(f"  Market Cap: {market_cap:,}" if market_cap else "  Market Cap: N/A")
    print(f"  ROE: {roe*100:.2f}%" if roe else "  ROE: N/A")
    print(f"  D/E: {de:.2f}" if de else "  D/E: N/A")

@app.command()
def var(ticker: str):
    """Get adaptive VaR estimate for a ticker"""
    ticker = ticker.upper()
    result = compute_live_var(ticker)
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    print(f"{ticker} — 1-Day VaR (99%): {result['var_99_1day']}%")
    print(f"Adapted alpha: {result['adapted_alpha']} (target: {result['target_alpha']})")
    print(result["interpretation"])

if __name__ == "__main__":
    load_config()
    app()