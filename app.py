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

if __name__ == "__main__":
    load_config()
    app()