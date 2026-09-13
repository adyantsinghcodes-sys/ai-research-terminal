import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    kite_key = os.getenv("KITE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"Config loaded — KITE_API_KEY set: {bool(kite_key)}, ANTHROPIC_API_KEY set: {bool(anthropic_key)}")