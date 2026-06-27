import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pathlib


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Gmt time' in df.columns:
        try:
            df['Gmt time'] = pd.to_datetime(df['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")
        except Exception:
            df['Gmt time'] = pd.to_datetime(df['Gmt time'], errors='coerce').fillna(pd.Timestamp.now())
    else:
        df['Gmt time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    return df.reset_index(drop=True)


def linear_schedule(initial_value: float, final_value: float):
    def lr_fn(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return lr_fn


def compute_max_drawdown(equity_series) -> float:
    arr = np.array(equity_series, dtype=float)
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    nonzero = running_max != 0
    drawdowns = np.zeros_like(arr)
    drawdowns[nonzero] = (running_max[nonzero] - arr[nonzero]) / running_max[nonzero]
    return float(np.max(drawdowns))


def compute_sharpe(equity_series) -> float:
    arr = np.array(equity_series, dtype=float)
    if arr.size < 2:
        return 0.0
    returns = np.diff(arr) / (arr[:-1] + 1e-9)
    std = np.std(returns)
    return float(np.mean(returns) / std) if std > 0 else 0.0


def print_trade_stats(trade_logs: list) -> None:
    profits = [t['profit_realized'] for t in trade_logs if 'profit_realized' in t]
    total = len(profits)
    if total == 0:
        print("No trades recorded.")
        return
    win_rate = sum(1 for p in profits if p > 0) / total
    holding_times = [t['holding_time'] for t in trade_logs if t.get('holding_time') is not None]
    print("=== TRADE STATISTICS ===")
    print(f"  Total trades      : {total}")
    print(f"  Win rate          : {win_rate:.2%}")
    print(f"  Mean trade profit : {np.mean(profits):.6f}")
    print(f"  Median profit     : {np.median(profits):.6f}")
    if holding_times:
        print(f"  Avg hold (bars)   : {np.mean(holding_times):.2f}")
    print("========================")


def plot_pnl(
    equities: list,
    buyhold: list,
    timestamps,
    output_dir: pathlib.Path,
) -> None:
    model_pnl = [e - equities[0] for e in equities]
    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        times = pd.to_datetime(timestamps)
        ax.plot(times, model_pnl, label='Model P&L')
        ax.plot(times, buyhold, label='Buy & Hold', linestyle='--')
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        ax.set_xlabel('Date')
    except Exception:
        x = np.arange(len(equities))
        ax.plot(x, model_pnl, label='Model P&L')
        ax.plot(x, buyhold, label='Buy & Hold', linestyle='--')
        ax.set_xlabel('Step')
    ax.set_ylabel('Profit')
    ax.set_title('Test P&L — Model vs Buy & Hold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "pnl_curve.png", dpi=150)
    plt.close()


def plot_weekly_pnl_distribution(weekly_pnl: list, output_dir: pathlib.Path) -> None:
    arr = np.array(weekly_pnl, dtype=float)
    if arr.size == 0:
        return
    mean, std = float(np.mean(arr)), float(np.std(arr))
    std = max(std, 1e-9)
    x = np.linspace(arr.min() - 0.05, arr.max() + 0.05, 800)
    y = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(arr, bins=30, density=True, alpha=0.6)
    ax.plot(x, y, linewidth=2)
    ax.axvline(0.0, linestyle='--', linewidth=1)
    ax.set_xlabel('Weekly P&L %')
    ax.set_ylabel('Density')
    ax.set_title(f'Weekly P&L Distribution  (mean={mean:.4f}, std={std:.4f})')
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "weekly_pnl_distribution.png", dpi=150)
    plt.close()
    pd.DataFrame({'weekly_pnl_pct': arr}).to_csv(output_dir / "weekly_pnl_pct.csv", index=False)
