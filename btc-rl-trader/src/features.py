import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True, fastmath=True)
def _higuchi_fd_numba(x: np.ndarray, kmax: int = 10) -> float:
    N = x.size
    if N < 3 or kmax < 2:
        return np.nan

    Lk = np.empty(kmax)
    for k in range(1, kmax + 1):
        Lm_sum = 0.0
        valid_m = 0
        for m in range(k):
            nm = (N - m - 1) // k
            if nm <= 0:
                continue
            Lm = 0.0
            for j in range(nm):
                Lm += abs(x[m + (j + 1) * k] - x[m + j * k])
            Lm = Lm * (N - 1) / (nm * k)
            Lm_sum += Lm
            valid_m += 1
        Lk[k - 1] = Lm_sum / valid_m if valid_m > 0 else np.nan

    valid = ~np.isnan(Lk) & (Lk > 0)
    ks = np.arange(1, kmax + 1)[valid]
    Lk = Lk[valid]
    if ks.size < 2:
        return np.nan

    xlog = -np.log(ks)
    ylog = np.log(Lk)
    n = xlog.size
    mean_x = np.sum(xlog) / n
    mean_y = np.sum(ylog) / n
    cov_xy = np.sum((xlog - mean_x) * (ylog - mean_y))
    var_x = np.sum((xlog - mean_x) ** 2)
    if var_x == 0:
        return np.nan

    slope = cov_xy / var_x
    return min(2.0, max(1.0, 2.0 - slope))


def rolling_higuchi_fd(series: pd.Series, window: int = 100, kmax: int = 10, min_periods: int = 20) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full_like(vals, np.nan)
    for i in range(window - 1, len(vals)):
        chunk = vals[i - window + 1: i + 1]
        if np.count_nonzero(~np.isnan(chunk)) >= min_periods:
            out[i] = _higuchi_fd_numba(chunk, kmax)
    return pd.Series(out, index=series.index)


@njit(cache=True, fastmath=True)
def _rolling_quantile_numba(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = arr.size
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(arr[i]):
            continue
        start = max(0, i - window)
        ref = arr[start:i]
        ref = ref[~np.isnan(ref)]
        if ref.size < min_periods:
            continue
        num_less = np.sum(ref < arr[i])
        num_equal = np.sum(ref == arr[i])
        frac = (num_less + 0.5 * num_equal) / ref.size
        pct = int(np.floor(frac * 100.0))
        out[i] = (min(100, max(0, pct)) / 50) - 1
    return out


def rolling_quantile(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    out = _rolling_quantile_numba(arr, window, min_periods)
    return pd.Series(out, index=series.index)


@njit(cache=True, fastmath=True)
def _rolling_trend_slope_numba(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.size
    out = np.full(n, np.nan)
    xi = np.arange(window).astype(np.float64)
    xi -= xi.mean()
    denom = np.sum(xi ** 2)
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        yi = y - np.mean(y)
        out[i] = np.sum(xi * yi) / denom
    return out


def rolling_trend_slope(series: pd.Series, window: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    out = _rolling_trend_slope_numba(arr, window)
    return pd.Series(out, index=series.index)


@njit(cache=True, fastmath=True)
def _rolling_trend_curvature_numba(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.size
    out = np.full(n, np.nan)
    xi = np.arange(window).astype(np.float64)
    xi -= np.mean(xi)
    ss_x = np.dot(xi, xi)
    xi2 = xi * xi
    xi2_centred = xi2 - np.mean(xi2)
    proj = np.dot(xi2_centred, xi) / ss_x
    xi2_orth = xi2_centred - proj * xi
    ss_xi2 = np.dot(xi2_orth, xi2_orth)
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        yi = y - np.mean(y)
        slope = np.dot(xi, yi) / ss_x
        residuals = yi - slope * xi
        out[i] = np.dot(xi2_orth, residuals) / ss_xi2
    return out


def rolling_trend_curvature(series: pd.Series, window: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    out = _rolling_trend_curvature_numba(arr, window)
    return pd.Series(out, index=series.index)


FEATURE_NAMES = [
    'returns', 'returns_5', 'returns_10',
    'returns_20', 'returns_d',
    'delta', 'delta_10', 'delta_30',
    'delta_100', 'volume', 'time',
    'slope_20', 'slope_30', 'slope_1000',
    'curve_20', 'curve_30', 'curve_1000',
    'vol_20', 'vol_1000', 'hfd_20', 'hfd_100',
    'quant_100', 'quant_200', 'quant_1000',
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ['close', 'open', 'volume', 'taker_buy_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['returns'] = ((df['close'] - df['close'].shift(1)) / df['close']).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df['returns_5'] = (df['close'] - df['close'].shift(5)) / df['close']
    df['returns_10'] = (df['close'] - df['close'].shift(10)) / df['close']
    df['returns_20'] = (df['close'] - df['close'].shift(20)) / df['close']
    df['returns_d'] = df['close'] / df.groupby(df['Gmt time'].dt.floor('D'))['close'].transform('first') - 1

    df['hfd_20'] = rolling_higuchi_fd(df['close'], window=20, kmax=10, min_periods=10)
    df['hfd_100'] = rolling_higuchi_fd(df['close'], window=100, kmax=10, min_periods=10)

    df['vol_20'] = (df['returns'] ** 2).rolling(window=20).sum().pow(0.5)
    df['vol_1000'] = (df['returns'] ** 2).rolling(window=1000).sum().pow(0.5)

    df['quant_100'] = rolling_quantile(df['close'], window=100, min_periods=10)
    df['quant_200'] = rolling_quantile(df['close'], window=200, min_periods=10)
    df['quant_1000'] = rolling_quantile(df['close'], window=1000, min_periods=10)

    df['slope_20'] = rolling_trend_slope(df['close'], window=20)
    df['slope_30'] = rolling_trend_slope(df['close'], window=30)
    df['slope_1000'] = rolling_trend_slope(df['close'], window=1000)

    df['curve_20'] = rolling_trend_curvature(df['close'], window=20)
    df['curve_30'] = rolling_trend_curvature(df['close'], window=30)
    df['curve_1000'] = rolling_trend_curvature(df['close'], window=1000)

    df['time'] = (df['Gmt time'] - df['Gmt time'].dt.floor('D')).dt.total_seconds() / 3600.0
    df['time'] = (df['time'] + 1) % 24

    df['delta'] = 2 * df['taker_buy_volume'] - df['volume']
    df['delta_10'] = df['delta'].rolling(window=10, min_periods=1).sum() / df['volume'].rolling(10).sum()
    df['delta_30'] = df['delta'].rolling(window=30, min_periods=1).sum() / df['volume'].rolling(30).sum()
    df['delta_100'] = df['delta'].rolling(window=100, min_periods=1).sum() / df['volume'].rolling(100).sum()

    s = df['delta']
    df['delta'] = np.sign(s) * np.log1p(s.abs())
    df['volume'] = np.log1p(df['volume'].clip(lower=0))

    df = df.fillna(0.0)
    return df
