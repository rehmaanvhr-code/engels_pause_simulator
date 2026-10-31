"""
Small macro simulation of Engels' pause.

A technology shock raises output per worker. Owners of scarce capital capture
the surplus until capital ownership spreads or policy steps in. The model is
calibrated to British data for 1760 to 1870 and then rerun at AI speed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "british_wages_output_1760_1870.csv")
OUT = os.path.join(HERE, "outputs")


def logistic(t, mid, width):
    return 1.0 / (1.0 + np.exp(-(t - mid) / width))


def run(years=110, shock=0.6, diffusion_years=60, owner_share0=0.02,
        ownership_spread=0.002, redistribution=0.0, bargaining=0.0,
        alpha=0.35, saving=0.3, depreciation=0.04, n_workers=1000,
        seed=0):
    """
    Run the model and return a DataFrame with one row per year.

    shock             total proportional rise in technology from the shock
    diffusion_years   years over which the new technology spreads
    owner_share0      fraction of the population that owns capital at the start
    ownership_spread  yearly growth of the owning share (capital diffusion)
    redistribution    share of profits transferred to workers each year
    bargaining        extra speed at which wages catch up to productivity
    """
    rng = np.random.default_rng(seed)
    n = n_workers
    L = 1.0
    A = 1.0
    # start from the steady state so nothing moves until the shock arrives
    K = (saving * alpha / depreciation) ** (1.0 / (1 - alpha))
    w = (1 - alpha) * A * K ** alpha
    owner_share = owner_share0
    holdings = rng.pareto(1.5, n) + 1.0
    rows = []
    for t in range(years):
        A = 1.0 + shock * logistic(t, diffusion_years / 2.0, diffusion_years / 8.0)
        Y = A * K ** alpha * L ** (1 - alpha)
        w_star = (1 - alpha) * Y / L
        # wages catch up faster when more people own capital and compete
        # for labour, and when workers have bargaining power
        speed = 0.4 * owner_share + bargaining
        speed = min(speed, 1.0)
        w = w + speed * (w_star - w)
        profits = Y - w * L
        transfer = redistribution * profits
        profits = profits - transfer
        wage_income = w * L + transfer
        K = K * (1 - depreciation) + saving * profits
        owner_share = min(owner_share + ownership_spread, 0.6)
        n_own = max(1, int(owner_share * n))
        owner_inc = profits * holdings[:n_own] / holdings[:n_own].sum()
        income = np.full(n, wage_income / n)
        income[:n_own] += owner_inc
        rows.append(dict(year=t, A=A, Y=Y, K=K, wage=wage_income / L,
                         profits=profits, owner_share=owner_share,
                         gini=gini(income)))
    df = pd.DataFrame(rows)
    df["wage_index"] = 100 * df["wage"] / df["wage"].iloc[0]
    df["output_index"] = 100 * (df["Y"] / L) / (df["Y"].iloc[0] / L)
    df["wage_share"] = df["wage"] * L / df["Y"]
    return df


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def pause_length(df, tol=0.02):
    """Years from the shock until the wage share is back within tol of its start."""
    share0 = df["wage_share"].iloc[0]
    gap = share0 - df["wage_share"]
    started = np.where(gap > tol)[0]
    if len(started) == 0:
        return 0
    start = started[0]
    after = np.where(gap.values[start:] <= tol)[0]
    if len(after) == 0:
        return len(df) - start
    return int(after[0])


def owner_surplus_share(df):
    """Share of the cumulative output gain that went to owners."""
    dY = (df["Y"] - df["Y"].iloc[0]).clip(lower=0).sum()
    dP = (df["profits"] - df["profits"].iloc[0]).clip(lower=0).sum()
    if dY == 0:
        return 0.0
    return float(min(dP / dY, 1.0))


def calibrate(hist):
    """Grid search over diffusion and ownership spread against the decade series."""
    best = None
    for diff in [60, 80, 100, 120]:
        for spread in [0.0005, 0.001, 0.002, 0.003, 0.004]:
            for shock in [0.3, 0.4, 0.5, 0.6, 0.8]:
                df = run(years=111, shock=shock, diffusion_years=diff,
                         ownership_spread=spread)
                sim = df.iloc[hist["year"] - 1760]
                err = ((sim["wage_index"].values - hist["real_wage_index"].values) ** 2).sum()
                err += ((sim["output_index"].values - hist["output_per_worker_index"].values) ** 2).sum()
                if best is None or err < best[0]:
                    best = (err, dict(shock=shock, diffusion_years=diff, ownership_spread=spread))
    return best[1]


def main():
    os.makedirs(OUT, exist_ok=True)
    hist = pd.read_csv(DATA)
    params = calibrate(hist)
    base = run(years=111, **params)
    base["calendar_year"] = 1760 + base["year"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(base["calendar_year"], base["output_index"], label="output per worker (simulated)")
    ax.plot(base["calendar_year"], base["wage_index"], label="real wage (simulated)")
    ax.scatter(hist["year"], hist["output_per_worker_index"], marker="s", label="output per worker (historical)")
    ax.scatter(hist["year"], hist["real_wage_index"], marker="o", label="real wage (historical)")
    ax.set_title("Engels' pause, Britain 1760 to 1870, model against data")
    ax.set_xlabel("year")
    ax.set_ylabel("index, 1760 = 100")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "engels_pause_baseline.png"), dpi=120)
    plt.close(fig)

    scenarios = {
        "historical pace": dict(diffusion_years=params["diffusion_years"],
                                ownership_spread=params["ownership_spread"]),
        "AI speed, no policy": dict(diffusion_years=10, ownership_spread=params["ownership_spread"]),
        "AI speed, broad ownership": dict(diffusion_years=10, ownership_spread=0.03),
        "AI speed, redistribution 20%": dict(diffusion_years=10,
                                             ownership_spread=params["ownership_spread"],
                                             redistribution=0.2),
        "AI speed, bargaining": dict(diffusion_years=10,
                                     ownership_spread=params["ownership_spread"],
                                     bargaining=0.15),
    }
    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, kw in scenarios.items():
        df = run(years=60, shock=params["shock"], **kw)
        axes[0].plot(df["year"], df["wage_index"], label=name)
        axes[1].plot(df["year"], df["gini"], label=name)
        summary.append(dict(scenario=name,
                            pause_years=pause_length(df),
                            owner_surplus_share=round(owner_surplus_share(df), 3),
                            wage_index_year_20=round(df["wage_index"].iloc[20], 1),
                            wage_index_year_59=round(df["wage_index"].iloc[59], 1),
                            gini_peak=round(df["gini"].max(), 3)))
    axes[0].set_title("Real wage index under each scenario")
    axes[0].set_xlabel("years since shock")
    axes[0].set_ylabel("index, start = 100")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Income Gini under each scenario")
    axes[1].set_xlabel("years since shock")
    axes[1].set_ylabel("Gini")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ai_scenarios.png"), dpi=120)
    plt.close(fig)

    out = pd.DataFrame(summary)
    out.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print("calibrated parameters:", params)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
