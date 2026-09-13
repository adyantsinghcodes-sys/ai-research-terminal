import typer
from config import load_config

app = typer.Typer()

@app.command()
def quote(ticker: str):
    """Get quote for a ticker"""
    print(f"quote called with {ticker}")

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