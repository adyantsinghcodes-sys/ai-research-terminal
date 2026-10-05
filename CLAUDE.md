# AI Research Terminal

Python CLI (Typer) that answers stock research questions, either as direct
commands or through an LLM with tool-calling. Repo:
`adyantsinghcodes-sys/ai-research-terminal`.

## Architecture

- `quote`, `fundamentals`, `var`, `exposure` — direct CLI commands over yfinance.
  All take `--market IN|US` (default IN).
- `ask` — one-shot natural-language query routed through Groq
  (`openai/gpt-oss-120b`) with tool-calling into the functions above.
  Takes `--market`, which sets `current_market`.
- `chat` — persistent session with conversation memory and an India/US market
  prompt at start.
- `SYSTEM_PROMPT` (shared by `ask` and `chat`) constrains the model to only
  state numbers returned by tools. This fixed a real hallucination bug where it
  invented sector benchmarks — don't weaken it. `system_prompt_for(market)`
  appends the session market; it must only ever append.
- `resolve_ticker()` keeps `.NS`/`.BO`, appends `.NS` for bare India tickers,
  leaves US tickers bare, and converts US share classes (`BRK.B` -> `BRK-B`).
- `get_exposure(market)` is the single implementation used by both `run_tool`
  and the `exposure` command. Portfolios are mocked (`PORTFOLIOS`, IN and US);
  Kite Connect integration is deferred. Returns `{"error": ...}` if any
  holding's download fails (yf.download does not raise on failure).

## Conventions

- Run `pytest` before proposing any commit. 23 tests, ~2s, no network.
- Tests mock yfinance and the Groq client via an autouse fixture that fails on
  any real call. Keep it that way — tests must run offline.
- Work on feature branches, merged fast-forward into `main`.
- For small edits, give a diff rather than rewriting the whole file.
- Test new logic in the REPL before wiring it into `app.py`.
- Stage files explicitly (`git add <file>`), never `git add .`.

## Known issues (priority order)

1. `compute_live_var` is a rolling empirical quantile with an ACI alpha update.
   Public claims were corrected to drop GARCH. Implementing it (e.g. filtered
   historical simulation: GARCH(1,1) + empirical quantile of standardised
   residuals) is optional and only worth it if it can be defended.
2. ACI parameters were tuned on index data (^NSEI/^GSPC), not individual
   stocks. Next step: backtest realised breach rates on 5–10 single stocks.
3. `test_app.py` has a comment asking that two key lists be kept in sync.
   Should be a shared `VAR_KEYS` constant.
4. Quote and fundamentals logic is still duplicated between `run_tool` and the
   CLI commands (exposure and VaR are already shared).
5. `get_exposure`: a ticker that downloads but is missing only the latest row
   yields NaN for that holding. Not handled yet.
6. In US mode, a foreign suffix (e.g. `VOD.L`) becomes `VOD-L`. Acceptable
   while scoped to IN/US; extend `EXCHANGE_SUFFIXES` if that changes.