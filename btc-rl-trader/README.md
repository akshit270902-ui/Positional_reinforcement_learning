# BTC Recurrent PPO Trading Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **recurrent reinforcement learning** trading agent for BTC/USDT (1-hour bars), built on:

- **RecurrentPPO** (LSTM policy) via `sb3-contrib`, enabling the agent to condition on hidden sequential state across steps.
- **Custom feature engineering** — fractal dimension (Higuchi FD), rolling quantile rank, trend slope/curvature, order-flow imbalance, and multi-horizon returns.
- **VecNormalize** observation normalization fitted strictly on training data, applied frozen at evaluation to prevent data leakage.
- **Walk-forward evaluation** on a held-out 20% test set, featuring an LSTM state-priming buffer before measuring performance.

---

## Motivation

Standard feedforward RL policies treat each timestep as independent, which fails to capture the strong path-dependencies inherent to financial markets (e.g., momentum, mean-reversion, and regime persistence). This system implements a recurrent policy with an LSTM hidden layer to integrate historical information over time implicitly. 

To complement the discrete reward topology and prevent state-blind optimization, we append real-time positional features directly to the observation vector. This allows the network to dynamically track its inventory, open exposure metrics, and historical execution trajectories.

### Key Design Parameters:
- **Higuchi Fractal Dimension:** Measures local price roughness to proxy shifting market regimes (trending vs. mean-reverting).
- **Rolling Quantile Rank:** Encodes where the current price sits in its historical distribution, normalized bounded to `[-1, 1]`.
- **Order-Flow Imbalance:** Captures localized net aggressive volume delta to model immediate directional pressure.
- **Trend Slope and Curvature:** Extracted via orthogonal polynomial regressions to isolate the first and second derivatives of price paths across multiple horizons.
- **LSTM Hidden State:** Bypasses explicit manual regime labeling by allowing the network to continuously maintain a latent representation of market microstructure states.

---

## Method

### Feature Construction & Mathematical Formulations

For each historical 1-hour bar, 24 raw features are computed and appended with 2 positional tracking attributes at inference time:

| Feature Category | Identifier | Description |
|---|---|---|
| **Multi-Horizon Returns** | `returns`, `returns_20` | Log price changes across varying lag structures |
| **Intraday Momentum** | `returns_d` | Log return calculated relative to daily session open |
| **Path Roughness** | `hfd_20`, `hfd_100` | Higuchi Fractal Dimension over 20-bar and 100-bar horizons |
| **Realized Volatility** | `vol_20`, `vol_1000` | Rolling standard deviation of log returns |
| **Distributional Sizing** | `quant_100/200/1000` | Rolling quantile rank, scaled uniformly to `[-1, 1]` |
| **Path Trajectory** | `slope_20/1000` | OLS linear trend slope across multiple lookbacks |
| **Path Acceleration** | `curve_20/1000` | Quadratic curvature (residual component from linear trend) |
| **Microstructure Signal** | `delta`, `delta_100` | Localized order-flow imbalance (Volume Delta) vectors |
| **Volume Dynamic** | `volume` | Absolute bar volume |
| **Temporal Context** | `time` | Hour of day transformed into a cyclic sine/cosine mapping |
| **Position Inventory** | `position` | Current agent state matrix: Long (`1`), Flat (`0`), Short (`-1`) |
| **Unrealized Exposure** | `position_return` | Floating PnL of current open trade position |

#### 1. Higuchi Fractal Dimension (HFD)
Computed by treating the partial price time-series as a curve stretching across time scales $k = 1, \dots, k_{max}$. The normalized path length $L_m(k)$ is defined as:
$$L_m(k) = \frac{1}{k} \left[ \left( \sum_{i=1}^{\lfloor \frac{N-m}{k} \rfloor} |X(m + ik) - X(m + (i-1)k)| \right) \frac{N-1}{\lfloor \frac{N-m}{k} \rfloor k} \right]$$
The fractal dimension $D$ is extracted via the least-squares linear regression slope of $\ln \langle L(k) \rangle$ against $\ln(1/k)$.

#### 2. Trend Curvature (Quadratic Polynomial Regression)
Isolated by fitting a second-order polynomial model across an orthogonal rolling horizon matrix:
$$y_t = \beta_0 + \beta_1 t + \beta_2 t^2 + \varepsilon_t$$
Where $\beta_1$ characterizes structural momentum (velocity) and $\beta_2$ characterizes trend acceleration/deceleration (curvature).

---

### Policy Architecture

- **Features Extractor:** Linear projection framework mapping raw observation matrices $\to 64$.
- **Recurrent Core:** 1-Layer LSTM network featuring a hidden size of 128 (completely decoupled Actor and Critic paths).
- **Network Heads:** Independent MLP structures `[128, 128]` for both Policy ($\pi$) and Value ($V$) estimation.
- **Optimization Vector:** Adam optimizer (`eps=1e-5`) governed by a linear learning rate decay schedule ($3 \times 10^{-4} \to 3 \times 10^{-5}$).

---

### Action Space & Friction Constraints

| Action Value | Structural Policy Mapping |
|:---:|---|
| `0` | **Flat:** Liquidate open exposure immediately / remain in cash |
| `1` | **Long:** Open new buy position / maintain active long hold |
| `2` | **Short:** Open new sell position / maintain active short hold |

- **Execution Model:** Order executions are processed at the bar boundary close. To account for market microstructural frictions, a standard exchange fee of **0.02% (2 bps)** is levied on each trade execution to reflect passive limit-order execution matching (parameters configurable inside `config.py`).

---

### Reward Mechanics

To prevent policy destabilization caused by aggressive continuous reward-shaping heuristics, the environment utilizes a transaction-isolated, sparse reward mapping evaluated exclusively at trade liquidation:

$$R_t = \begin{cases} 
\text{Side} \cdot \left( \frac{S_{\text{close}} - S_{\text{entry}}}{S_{\text{entry}}} \right) - \mathcal{C} & \text{if position closes at step } t \\
0 & \text{if position is held or agent is flat}
\end{cases}$$

Where $\text{Side} \in \{-1, 1\}$ and $\mathcal{C}$ is the execution friction penalty matrix.

---

### Walk-Forward Evaluation Rigor

- **Training Partition:** Initial 80% contiguous slice of historical records (processed after a 12-month warm-up block to settle slow-decay features).
- **Out-of-Sample Testing:** Final 20% validation window, initialized with a **1,000-bar priming buffer** to completely populate and warm up the LSTM's initial hidden state vectors before recording strategy returns.
- **Normalization Stability:** `VecNormalize` running statistics are computed and locked on the training set, and applied as a frozen transformation during testing.

---

## Quickstart

```bash
git clone [https://github.com/](https://github.com/)<your-handle>/btc-rl-trader.git
cd btc-rl-trader
pip install -r requirements.txt
