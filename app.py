import typer
from config import load_config
import yfinance as yf

app = typer.Typer()

@app.command()
def quote(ticker: str):
    """Get quote for a ticker"""
    ticker = ticker.upper()
    t = yf.Ticker(f"{ticker}.NS")
    data = t.history(period="2d")

    if data.empty:
        print(f"No data found for {ticker}")
        return

    latest = data["Close"].iloc[-1]
    previous = data["Close"].iloc[-2]
    change = latest - previous
    pct_change = (change / previous) * 100

    print(f"{ticker}: {latest:.2f}  ({change:+.2f}, {pct_change:+.2f}%)")
  

@app.command()
def exposure():
    """Show portfolio exposure"""
    print("exposure called")

@app.command()
def ask(query: str):
    """Ask the AI analyst a question"""
    print(f"ask called with: {query}")

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