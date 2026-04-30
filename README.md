# ml_analysis_options

Options-trading variant of [ml_analysis](https://github.com/vuoalbert/ml_analysis).
Reuses 70% of the validated stock-trading infrastructure, replaces the execution
layer with options trading.

## Strategy

- **Entry signal**: same LightGBM model as ml_analysis (66 features incl. multi-timeframe).
  When `p_up ≥ 0.55`, go long → buy a call. When `p_down ≥ 0.55`, go short → buy a put.
- **Contract selection**: 0DTE ATM SPY option by default (most liquid)
- **Sizing**: risk-pct of equity (1% by default), capped at 10 contracts max
- **Exit**: premium-based (stop at -50%, target at +100%), plus theta-protection
  (force exit in last hour of session if not in profit), plus EOD-flat
- **State recovery**: skipped for options in this MVP (options-mode boot starts flat)

## What's the same as ml_analysis

| Component | Status |
|---|---|
| Entry model artifact (`artifacts/latest/`) | Identical |
| V2 exit predictors (`research_v2_exit_tight/`) | Identical, but unused in options mode |
| Feature pipeline (`features/build.py` + `feature_extensions.py`) | Identical |
| Data assembly (`data_pull/`) | Identical |
| Daily DD kill, halt flag, EOD-flat, heartbeat, journal | Identical |
| Dashboard (`ui/`) | Identical layout, options-mode labels TODO |
| Discord wiring | Reused, simpler embeds for options |

## What's different

| Component | Change |
|---|---|
| **`live/options.py`** | NEW. Strike selection, OCC symbol formatting, premium quotes, options-specific exit logic |
| **`live/loop.py`** | Added `mode` dispatch in `__init__`, `_plan_options_entry`, `_submit_option`, `_flat_options`. Stock-mode path preserved as legacy. |
| **`configs/v1.yaml`** | New `strategy.mode: options` block with options-specific parameters |
| **Position struct** | Added `option_symbol`, `option_side`, `entry_premium`, `stop_pct`, `target_pct` fields |

## Switching modes

Edit `configs/v1.yaml`:
```yaml
strategy:
  mode: options    # or "stocks"
  options:
    moneyness: atm
    expiration: same_day
    risk_pct_per_trade: 0.01
    max_qty_contracts: 10
    stop_pct: 0.50
    target_pct: 1.00
    theta_protect_mins: 60
    underlying: SPY
```

## Status (this is MVP, not production)

**Working:**
- ✓ Module structure, config dispatch
- ✓ Entry model loads, predicts (uses same v3_mtf as ml_analysis)
- ✓ Contract selection from Alpaca chain
- ✓ OCC symbol formatting
- ✓ Premium-based qty sizing
- ✓ Submit option market order
- ✓ Per-tick exit check (premium-based stop/target/theta)
- ✓ Force-flat at EOD
- ✓ Discord alerts (basic embeds)

**Not yet built (TODO):**
- ✗ Options-specific backtest harness (would need historical option chain data)
- ✗ Greeks-aware position sizing (currently just premium-based)
- ✗ Spread strategies (verticals, iron condors)
- ✗ State recovery for boot-time recovered options positions
- ✗ Dashboard updates for option-position rendering (still shows stock-style position card)
- ✗ Strike-selection backtest sweep (ATM vs OTM vs ITM optimization)

**Risks before live deploy:**
- Alpaca options approval Level 2 required (apply via Alpaca dashboard, takes 1-3 business days)
- Need historical option chain data for honest backtest (not included in Alpaca free tier)
- 0DTE options have huge theta — small mistakes in exit timing can cost 30-50% of premium
- Win rates on options strategies typically 30-40%, vs 60-70% for the stock strategy
- Account drawdowns are bigger in % terms even with smaller absolute $

## Running it

### Local smoke test
```bash
python -m live.loop --once --force --dry-run
```
Should log `strategy mode: OPTIONS` on init. The `--dry-run` flag prevents actual order submission.

### Docker
```bash
docker compose build
docker compose up -d
docker compose logs -f loop
```
Dashboard appears on port 8503 (configured in `docker-compose.yml`).

### Side-by-side with ml_analysis
This repo is intentionally compatible with running alongside the original `ml_analysis` stock bot on the same VM (different container names, different host port). They share `cache/` if you symlink, otherwise each pulls bars independently.

## Provenance

- Forked from ml_analysis at commit `1cbb599` (post v3_mtf-live ship)
- Same author, same data, same model, different execution
