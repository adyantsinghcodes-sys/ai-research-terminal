import os
import typer
import json
from groq import Groq
from config import load_config
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

app = typer.Typer()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
def ask(query: str):
    """Ask the AI analyst a question"""
    messages = [{"role": "user", "content": query}]

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
    """Show portfolio exposure"""
    print("exposure called")


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


if __name__ == "__main__":
    load_config()
    app()