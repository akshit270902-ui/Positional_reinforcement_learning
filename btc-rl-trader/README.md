# BTC Recurrent PPO Trading Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **recurrent reinforcement learning** trading agent for BTC/USDT (1-hour bars), built on:

- **RecurrentPPO** (LSTM policy) via `sb3-contrib`, enabling the agent to condition on hidden sequential state across steps
- **Custom feature engineering** — fractal dimension (Higuchi FD), rolling quantile rank, trend slope/curvature, order-flow imbalance, and multi-horizon returns
- **VecNormalize** observation normalisation fitted on training data, applied frozen at evaluation
- **Walk-forward evaluation** on a held-out 20% test set, with a buffer warm-up period to prime LSTM state before measuring performance

---

## Motivation

Standard feedforward RL policies treat each timestep as independent. Financial markets exhibit strong path-dependency: momentum, mean-reversion, and regime persistence all require the agent to integrate information over time. A recurrent policy with LSTM hidden state naturally captures this without manual feature engineering of lookback windows.

Key design choices:

- **Higuchi fractal dimension** measures local price roughness — a proxy for regime type (trending vs. noisy)
- **Rolling quantile rank** encodes where the current price sits in its recent distribution, normalised to `[-1, 1]`
- **Order-flow imbalance** (`2 * taker_buy_volume - volume`) proxies directional pressure; log-transformed to compress outliers
- **Trend slope and curvature** from orthogonal polynomial regression capture first and second derivatives of price over multiple horizons
- **LSTM hidden state** allows the policy to implicitly track regime without explicit regime labels

---

## Method

### Feature Construction

For each bar, 24 raw features are computed:

| Feature | Description |
|---|---|
| `returns`, `returns_5/10/20` | Log price changes at multiple horizons |
| `returns_d` | Intraday return since daily open |
| `hfd_20`, `hfd_100` | Higuchi fractal dimension over 20 and 100 bars |
| `vol_20`, `vol_1000` | Rolling realised volatility |
| `quant_100/200/1000` | Rolling quantile rank, scaled to `[-1, 1]` |
| `slope_20/30/1000` | OLS trend slope at three horizons |
| `curve_20/30/1000` | Quadratic curvature (residual from linear trend) |
| `delta`, `delta_10/30/100` | Order-flow imbalance at spot and rolling horizons |
| `volume` | Log-transformed bar volume |
| `time` | Hour of day (cyclic, 0–23) |

Two positional features are appended at inference time: current `position` (`{-1, 0, 1}`) and unrealised `position_return`.

### Policy Architecture

- **Features extractor**: linear projection from raw observation dim → 64
- **LSTM**: 1 layer, hidden size 64, per-actor (critic has its own LSTM)
- **Policy / value heads**: `[128, 128]` MLP each
- **Optimiser**: Adam, `eps=1e-5`, linear LR schedule `3e-4 → 3e-5`

### Action Space

| Action | Meaning |
|---|---|
| `0` | Flat (close any open position) |
| `1` | Long |
| `2` | Short |

Transitions are immediate at the current bar's close price. Commission is set to zero by default (configurable in `config.py`).

### Reward

Realised PnL as a fraction of entry price, on trade close. No shaping beyond the raw profit signal.

### Walk-Forward Evaluation

- Train: first 80% of data (after 12-month warm-up for slow features)
- Test: last 20%, preceded by a 1000-bar buffer for LSTM state priming
- VecNormalize statistics are frozen during evaluation

---

## Quickstart

```bash
git clone https://github.com/<your-handle>/btc-rl-trader.git
cd btc-rl-trader
pip install -r requirements.txt
```

Place `BTCUSDT_1h.csv` in `data/raw/`. Then:

```bash
python scripts/train.py
python scripts/evaluate.py
```

---

## Results

| Metric | Value |
|---|---|
| Total trades (test) | — |
| Trade win rate | — |
| Mean trade profit | — |
| Avg holding time (bars) | — |
| Weekly P&L mean / std | — |

> Fill in after running evaluation. Weekly P&L is expressed as a fraction of the week-start reference price.

---

## Project Structure

```
src/features.py      — all feature engineering (Higuchi FD, quantile rank, slope/curvature, order flow)
src/env.py           — TradingEnv (gymnasium), observation and step logic
src/policy.py        — CustomFeaturesExtractor and policy_kwargs
src/utils.py         — data loading, metrics, plotting helpers
config.py            — all hyperparameters in one place
scripts/train.py     — training entry point
scripts/evaluate.py  — walk-forward test evaluation
tests/               — unit tests for features and environment
```

---

## Dependencies

See `requirements.txt`. Core: `stable-baselines3`, `sb3-contrib`, `gymnasium`, `torch`, `numba`, `pandas`, `numpy`, `matplotlib`.

---

## License

MIT
