# AI Research Terminal

Python CLI (Typer) that answers stock research questions, either as direct
commands or through an LLM with tool-calling. Repo:
`adyantsinghcodes-sys/ai-research-terminal`.

## Architecture

- `quote`, `fundamentals`, `var`, `exposure` — direct CLI commands over yfinance.
- `ask` — one-shot natural-language query routed through Groq
  (`openai/gpt-oss-120b`) with tool-calling into the functions above.
- `chat` — persistent session with conversation memory and an India/US market
  prompt at start.
- `SYSTEM_PROMPT` (shared by `ask` and `chat`) constrains the model to only
  state numbers returned by tools. This fixed a real hallucination bug where it
  invented sector benchmarks — don't weaken it.
- `resolve_ticker()` appends `.NS` for India, leaves bare for US.
- `exposure` uses a mocked portfolio; Kite Connect integration is deferred.

## Conventions

- Run `pytest` before proposing any commit. 18 tests, ~2s, no network.
- Tests mock yfinance via an autouse fixture that fails on any real call.
  Keep it that way — tests must run offline.
- Work on feature branches, merged fast-forward into `main`.
- For small edits, give a diff rather than rewriting the whole file.
- Test new logic in the REPL before wiring it into `app.py`.

## Known issues (priority order)

1. The project description claims GARCH + ACI, but `compute_live_var` is a
   rolling empirical quantile with an ACI alpha update — there is no GARCH.
   Either correct the claim or implement it.
2. README doesn't document how to run the tests or that they run offline.
3. `get_exposure` hardcodes `.NS` and ignores `current_market`, so US chat mode
   silently returns Indian holdings. The logic is also duplicated between
   `run_tool` and the `exposure` command.
4. `test_app.py` has a comment asking that two key lists be kept in sync.
   Should be a shared `VAR_KEYS` constant.
5. ACI parameters were tuned on index data (^NSEI/^GSPC), not individual
   stocks. Noted in the README.