"""
Portfolio Optimizer — Markowitz Efficient Frontier
====================================================
Given a basket of tickers, this script:
  1. Pulls historical price data (yfinance)
  2. Computes expected returns & covariance matrix
  3. Runs Monte Carlo simulation of random portfolios
  4. Solves for the Max Sharpe Ratio and Min Variance portfolios
     using constrained optimization (scipy)
  5. Plots the efficient frontier

Usage:
    python portfolio_optimizer.py --tickers AAPL MSFT GOOGL AMZN NVDA --years 3
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

RISK_FREE_RATE = 0.045  # approx annualized T-bill rate, adjust as needed


def fetch_data(tickers, years):
    """Download adjusted close prices and compute daily returns."""
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw = raw.dropna(how="all")
    returns = raw.pct_change().dropna()
    return returns


def portfolio_stats(weights, mean_returns, cov_matrix):
    """Annualized return, volatility, and Sharpe ratio for a given weight vector."""
    ann_return = np.dot(weights, mean_returns) * 252
    ann_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol
    return ann_return, ann_vol, sharpe


def neg_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_stats(weights, mean_returns, cov_matrix)[2]


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_stats(weights, mean_returns, cov_matrix)[1]


def optimize_portfolio(mean_returns, cov_matrix, objective):
    n = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(n))  # long-only, no leverage
    init_guess = np.array([1 / n] * n)
    result = minimize(objective, init_guess, args=args, method="SLSQP",
                       bounds=bounds, constraints=constraints)
    return result.x


def monte_carlo_simulation(mean_returns, cov_matrix, n_portfolios=8000):
    n_assets = len(mean_returns)
    results = np.zeros((3, n_portfolios))
    weights_record = []
    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        weights_record.append(w)
        ret, vol, sharpe = portfolio_stats(w, mean_returns, cov_matrix)
        results[0, i], results[1, i], results[2, i] = ret, vol, sharpe
    return results, weights_record


def main():
    parser = argparse.ArgumentParser(description="Markowitz Portfolio Optimizer")
    parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT GOOGL")
    parser.add_argument("--years", type=int, default=3, help="Years of historical data")
    parser.add_argument("--simulations", type=int, default=8000)
    parser.add_argument("--out", type=str, default="efficient_frontier.png")
    args = parser.parse_args()

    print(f"Fetching {args.years}y of data for: {', '.join(args.tickers)}")
    returns = fetch_data(args.tickers, args.years)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    max_sharpe_w = optimize_portfolio(mean_returns, cov_matrix, neg_sharpe)
    min_vol_w = optimize_portfolio(mean_returns, cov_matrix, portfolio_volatility)

    ms_ret, ms_vol, ms_sharpe = portfolio_stats(max_sharpe_w, mean_returns, cov_matrix)
    mv_ret, mv_vol, mv_sharpe = portfolio_stats(min_vol_w, mean_returns, cov_matrix)

    print("\n=== Max Sharpe Portfolio ===")
    for t, w in zip(args.tickers, max_sharpe_w):
        print(f"  {t}: {w:.2%}")
    print(f"  Expected Return: {ms_ret:.2%} | Volatility: {ms_vol:.2%} | Sharpe: {ms_sharpe:.2f}")

    print("\n=== Min Volatility Portfolio ===")
    for t, w in zip(args.tickers, min_vol_w):
        print(f"  {t}: {w:.2%}")
    print(f"  Expected Return: {mv_ret:.2%} | Volatility: {mv_vol:.2%} | Sharpe: {mv_sharpe:.2f}")

    print(f"\nRunning Monte Carlo simulation ({args.simulations} portfolios)...")
    results, _ = monte_carlo_simulation(mean_returns, cov_matrix, args.simulations)

    # Plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(results[1], results[0], c=results[2], cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(scatter, label="Sharpe Ratio")
    plt.scatter(ms_vol, ms_ret, c="red", marker="*", s=400, label="Max Sharpe")
    plt.scatter(mv_vol, mv_ret, c="blue", marker="*", s=400, label="Min Volatility")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title(f"Efficient Frontier — {', '.join(args.tickers)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()