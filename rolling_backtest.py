"""
Rolling Backtest — Out-of-Sample Portfolio Optimization
=========================================================
Naive Markowitz optimization overfits badly when applied in-sample: the
"optimal" weights are fit to the exact historical period they're then
evaluated on, which flatters the result. This script fixes that by:

  1. Re-optimizing the Max Sharpe portfolio on a trailing window
     (e.g. the past 12 months)
  2. Holding those weights fixed and applying them to the *next*
     period's actual (unseen) returns
  3. Rolling forward and repeating (walk-forward validation)
  4. Comparing the resulting equity curve against:
       - An equal-weight portfolio (naive baseline)
       - A benchmark (e.g. SPY)

This produces a much more honest picture of whether mean-variance
optimization actually adds value out-of-sample.

Usage:
    python rolling_backtest.py --tickers AAPL MSFT GOOGL AMZN NVDA \
        --years 5 --window 12 --rebalance 3 --benchmark SPY
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from portfolio_optimizer import (
    fetch_data,
    optimize_portfolio,
    neg_sharpe,
    portfolio_stats,
    RISK_FREE_RATE,
)


def rolling_backtest(returns, window_months, rebalance_months):
    """
    Walk-forward test: optimize on trailing `window_months` of data,
    hold weights for `rebalance_months`, then re-optimize.
    Returns a Series of portfolio daily returns (out-of-sample only).
    """
    monthly_index = pd.date_range(returns.index.min(), returns.index.max(), freq="MS")
    portfolio_returns = pd.Series(dtype=float)
    weights_log = []

    for i in range(window_months, len(monthly_index) - rebalance_months, rebalance_months):
        train_start = monthly_index[i - window_months]
        train_end = monthly_index[i]
        test_start = train_end
        test_end = monthly_index[min(i + rebalance_months, len(monthly_index) - 1)]

        train = returns.loc[train_start:train_end]
        test = returns.loc[test_start:test_end]

        if len(train) < 20 or len(test) < 1:
            continue

        mean_returns = train.mean()
        cov_matrix = train.cov()
        weights = optimize_portfolio(mean_returns, cov_matrix, neg_sharpe)
        weights_log.append((test_start, dict(zip(returns.columns, weights))))

        # Apply held weights to next period's ACTUAL (unseen) returns
        period_portfolio_returns = test @ weights
        portfolio_returns = pd.concat([portfolio_returns, period_portfolio_returns])

    return portfolio_returns.sort_index(), weights_log


def equal_weight_returns(returns):
    n = returns.shape[1]
    weights = np.array([1 / n] * n)
    return returns @ weights


def fetch_benchmark(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    # yfinance sometimes returns a 1-column DataFrame instead of a Series
    # even for a single ticker (depends on version) — normalize to a Series.
    if isinstance(raw, pd.DataFrame):
        raw = raw.squeeze("columns")
    return raw.pct_change().dropna()


def annualized_stats(daily_returns):
    daily_returns = pd.Series(np.asarray(daily_returns).ravel(), index=pd.Series(daily_returns).index)
    ann_return = float(daily_returns.mean() * 252)
    ann_vol = float(daily_returns.std() * np.sqrt(252))
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1).min()
    return {
        "Annualized Return": ann_return,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown,
        "Total Return": cumulative.iloc[-1] - 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Rolling out-of-sample portfolio backtest")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--years", type=int, default=5, help="Total history to pull")
    parser.add_argument("--window", type=int, default=12, help="Trailing training window (months)")
    parser.add_argument("--rebalance", type=int, default=3, help="Rebalance frequency (months)")
    parser.add_argument("--benchmark", type=str, default="SPY")
    parser.add_argument("--out", type=str, default="rolling_backtest.png")
    args = parser.parse_args()

    print(f"Fetching {args.years}y of data for: {', '.join(args.tickers)}")
    returns = fetch_data(args.tickers, args.years)

    print(f"Running walk-forward backtest (train={args.window}mo, rebalance every {args.rebalance}mo)...")
    opt_returns, weights_log = rolling_backtest(returns, args.window, args.rebalance)

    eq_returns = equal_weight_returns(returns).loc[opt_returns.index]

    start, end = opt_returns.index.min(), opt_returns.index.max()
    bench_returns = fetch_benchmark(args.benchmark, start, end)
    bench_returns = bench_returns.reindex(opt_returns.index).dropna()
    common_idx = opt_returns.index.intersection(bench_returns.index)
    opt_returns_c = opt_returns.loc[common_idx]
    eq_returns_c = eq_returns.loc[common_idx]
    bench_returns_c = bench_returns.loc[common_idx]

    print("\n=== Out-of-Sample Performance ===")
    for name, series in [
        ("Optimized (Max Sharpe, rolling)", opt_returns_c),
        ("Equal Weight", eq_returns_c),
        (args.benchmark, bench_returns_c),
    ]:
        stats = annualized_stats(series)
        print(f"\n{name}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.2%}" if k != "Sharpe" else f"  {k}: {v:.2f}")

    # Plot cumulative equity curves
    plt.figure(figsize=(11, 6))
    for name, series in [
        ("Optimized (rolling Max Sharpe)", opt_returns_c),
        ("Equal Weight", eq_returns_c),
        (args.benchmark, bench_returns_c),
    ]:
        cumulative = (1 + series).cumprod()
        plt.plot(cumulative.index, cumulative.values, label=name, linewidth=2)

    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.title(f"Out-of-Sample Backtest: Rolling Optimization vs. Benchmarks\n"
              f"({', '.join(args.tickers)})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved equity curve chart to {args.out}")

    print("\nNote: if the optimized line does NOT clearly beat equal-weight,")
    print("that's a legitimate and common finding — write it up. It shows you")
    print("understand mean-variance optimization's real-world limitations")
    print("(estimation error, overfitting to trailing windows), not just how")
    print("to call scipy.optimize.")


if __name__ == "__main__":
    main()