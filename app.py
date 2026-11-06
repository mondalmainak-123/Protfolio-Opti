import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

st.set_page_config(
    page_title="MVP vs HRP Portfolio Optimization",
    layout="wide"
)

st.title("NIFTY 50 Portfolio Optimization")
st.subheader("Minimum Variance Portfolio (MVP) vs Hierarchical Risk Parity (HRP)")

# ------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------

DEFAULT_STOCKS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "RELIANCE.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "INFY.NS",
    "KOTAKBANK.NS",
    "M&M.NS"
]

with st.sidebar:
    st.header("Portfolio Settings")

    stock_input = st.text_area(
        "NIFTY 50 stock symbols (one per line)",
        value="\n".join(DEFAULT_STOCKS),
        height=250
    )

    stocks = list(dict.fromkeys(
        x.strip().upper()
        for x in stock_input.replace(",", "\n").splitlines()
        if x.strip()
    ))

    rf = st.number_input(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.0
    ) / 100

    st.markdown("---")
    st.write("Training period")
    train_start = st.date_input(
        "Training start",
        pd.Timestamp("2025-01-01")
    )
    train_end = st.date_input(
        "Training end",
        pd.Timestamp("2026-04-30")
    )

    st.markdown("---")
    st.write("Testing period")
    test_start = st.date_input(
        "Testing start",
        pd.Timestamp("2026-05-01")
    )
    test_end = st.date_input(
        "Testing end",
        pd.Timestamp("2026-10-31")
    )

    run = st.button(
        "Run Portfolio Optimization",
        type="primary",
        use_container_width=True
    )

# ------------------------------------------------
# 2. DOWNLOAD CLOSE PRICE DATA
# ------------------------------------------------

def download_data(stocks, start, end):

    data = yf.download(
        stocks,
        start=str(start),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column"
    )

    if data.empty:
        raise ValueError("No price data received from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data.xs("Close", axis=1, level=-1)
    else:
        prices = data[["Close"]]
        prices.columns = stocks[:1]

    prices.columns = [str(x).upper() for x in prices.columns]

    missing = [
        s for s in stocks
        if s not in prices.columns or prices[s].dropna().empty
    ]

    if missing:
        raise ValueError(
            "Missing stock data: " + ", ".join(missing)
        )

    prices = prices[stocks]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index().dropna(how="any")

    if len(prices) < 60:
        raise ValueError("Insufficient common price observations.")

    return prices

# ------------------------------------------------
# 3. MVP OPTIMIZATION
# ------------------------------------------------

def calculate_mvp(returns):

    cov = returns.cov().to_numpy(copy=True)

    # Ensure numerical symmetry
    cov = (cov + cov.T) / 2

    n = cov.shape[0]

    initial_weights = np.ones(n) / n

    def portfolio_variance(weights):
        return float(weights @ cov @ weights)

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        }
    ]

    bounds = [(0, 1)] * n

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12
        }
    )

    if not result.success:
        raise ValueError(result.message)

    weights = np.clip(result.x, 0, 1)

    weights /= weights.sum()

    return pd.Series(
        weights,
        index=returns.columns,
        name="MVP"
    )

# ------------------------------------------------
# 4. HRP OPTIMIZATION
# ------------------------------------------------

def cluster_variance(cov, cluster):

    sub_cov = cov.loc[cluster, cluster]

    inv_var = 1 / np.diag(sub_cov.values).clip(min=1e-12)

    weights = inv_var / inv_var.sum()

    return weights @ sub_cov.values @ weights


def calculate_hrp(returns):

    # Explicitly create independent, writable arrays
    cov = returns.cov().copy(deep=True)
    corr = returns.corr().copy(deep=True)

    # Clip correlations
    corr = corr.clip(-1, 1)

    # Calculate correlation distance using a writable NumPy array
    corr_array = corr.to_numpy(copy=True)

    distance_array = np.sqrt(
        np.maximum(0, (1 - corr_array) / 2)
    )

    # Set diagonal to zero
    np.fill_diagonal(distance_array, 0.0)

    # Hierarchical clustering
    condensed = squareform(
        distance_array,
        checks=False
    )

    tree = linkage(
        condensed,
        method="single",
        optimal_ordering=True
    )

    ordered = list(
        returns.columns[leaves_list(tree)]
    )

    weights = pd.Series(
        1.0,
        index=ordered
    )

    clusters = [ordered]

    while clusters:

        next_clusters = []

        for cluster in clusters:

            if len(cluster) <= 1:
                continue

            split = len(cluster) // 2

            left = cluster[:split]
            right = cluster[split:]

            var_left = cluster_variance(cov, left)
            var_right = cluster_variance(cov, right)

            alpha = 1 - var_left / (
                var_left + var_right
            )

            weights[left] *= alpha
            weights[right] *= (1 - alpha)

            next_clusters.extend([left, right])

        clusters = next_clusters

    weights = weights.reindex(returns.columns)

    weights /= weights.sum()

    weights.name = "HRP"

    return weights
# ------------------------------------------------
# 5. PERFORMANCE METRICS
# ------------------------------------------------

def calculate_metrics(returns, rf):

    returns = returns.dropna()

    wealth = (1 + returns).cumprod()

    total_return = wealth.iloc[-1] - 1

    years = len(returns) / 252

    cagr = (
        wealth.iloc[-1] ** (1 / years) - 1
        if years > 0 and wealth.iloc[-1] > 0
        else np.nan
    )

    annual_risk = returns.std() * np.sqrt(252)

    sharpe = (
        (returns.mean() - rf / 252)
        / returns.std()
        * np.sqrt(252)
        if returns.std() > 0
        else np.nan
    )

    drawdown = wealth / wealth.cummax() - 1

    return {
        "Total Return": total_return,
        "Annualized Return": cagr,
        "Annualized Risk": annual_risk,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": drawdown.min()
    }

# ------------------------------------------------
# 6. MAIN ANALYSIS
# ------------------------------------------------

if run:

    if len(stocks) < 2:
        st.error("Please enter at least two stocks.")
        st.stop()

    if train_start >= train_end or test_start > test_end:
        st.error("Invalid date range.")
        st.stop()

    if train_end >= test_start:
        st.error("Training must end before testing starts.")
        st.stop()

    try:

        with st.spinner("Downloading stock prices and optimizing portfolios..."):

            prices = download_data(
                stocks,
                train_start,
                test_end
            )

            # Use Close prices only
            returns = prices.pct_change().dropna()

            train_returns = returns.loc[
                (returns.index.date >= train_start)
                & (returns.index.date <= train_end)
            ]

            test_returns = returns.loc[
                (returns.index.date >= test_start)
                & (returns.index.date <= test_end)
            ]

            if len(train_returns) < 60 or len(test_returns) < 20:
                raise ValueError("Insufficient training or testing observations.")

            # Optimize using training data only
            mvp_weights = calculate_mvp(train_returns)

            hrp_weights = calculate_hrp(train_returns)

            # Fixed weights during testing
            portfolio_returns = pd.DataFrame({

                "MVP": test_returns @ mvp_weights,

                "HRP": test_returns @ hrp_weights

            }).dropna()

            cumulative_returns = (
                1 + portfolio_returns
            ).cumprod()

            # Performance statistics
            mvp_metrics = calculate_metrics(
                portfolio_returns["MVP"], rf
            )

            hrp_metrics = calculate_metrics(
                portfolio_returns["HRP"], rf
            )

            metrics = pd.DataFrame(
                [mvp_metrics, hrp_metrics],
                index=["MVP", "HRP"]
            )

        st.success("Portfolio optimization completed!")

        # ------------------------------------------------
        # 7. PERFORMANCE SUMMARY
        # ------------------------------------------------

        st.header("Out-of-Sample Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "MVP Return",
            f"{mvp_metrics['Total Return']:.2%}"
        )

        col2.metric(
            "HRP Return",
            f"{hrp_metrics['Total Return']:.2%}"
        )

        col3.metric(
            "MVP Sharpe Ratio",
            f"{mvp_metrics['Sharpe Ratio']:.3f}"
        )

        col4.metric(
            "HRP Sharpe Ratio",
            f"{hrp_metrics['Sharpe Ratio']:.3f}"
        )

        st.subheader("Performance Comparison")

        display_metrics = metrics.copy()

        for col in [
            "Total Return",
            "Annualized Return",
            "Annualized Risk",
            "Maximum Drawdown"
        ]:
            display_metrics[col] = display_metrics[col].map(
                lambda x: f"{x:.2%}"
            )

        display_metrics["Sharpe Ratio"] = (
            display_metrics["Sharpe Ratio"].map(
                lambda x: f"{x:.3f}"
            )
        )

        st.dataframe(
            display_metrics,
            use_container_width=True
        )

        # ------------------------------------------------
        # 8. CUMULATIVE RETURN PLOT
        # ------------------------------------------------

        st.header("Portfolio Return Analysis")

        fig1 = go.Figure()

        for strategy in cumulative_returns.columns:

            fig1.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[strategy],
                    mode="lines",
                    name=strategy
                )
            )

        fig1.update_layout(
            title="Cumulative Portfolio Growth",
            xaxis_title="Date",
            yaxis_title="Growth of ₹1",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # ------------------------------------------------
        # 9. RISK COMPARISON
        # ------------------------------------------------

        rolling_risk = (
            portfolio_returns
            .rolling(21)
            .std() * np.sqrt(252)
        )

        fig2 = px.line(
            rolling_risk,
            title="21-Day Rolling Annualized Portfolio Risk",
            labels={
                "value": "Annualized Volatility",
                "index": "Date"
            }
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ------------------------------------------------
        # 10. RETURN COMPARISON
        # ------------------------------------------------

        fig3 = px.bar(
            metrics.reset_index(),
            x="index",
            y="Annualized Return",
            title="Annualized Return Comparison",
            labels={
                "index": "Portfolio Strategy",
                "Annualized Return": "CAGR"
            }
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # ------------------------------------------------
        # 11. SHARPE RATIO COMPARISON
        # ------------------------------------------------

        fig4 = px.bar(
            metrics.reset_index(),
            x="index",
            y="Sharpe Ratio",
            title="Sharpe Ratio Comparison",
            labels={
                "index": "Portfolio Strategy",
                "Sharpe Ratio": "Sharpe Ratio"
            }
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        # ------------------------------------------------
        # 12. PORTFOLIO ALLOCATIONS
        # ------------------------------------------------

        st.header("Portfolio Asset Allocations")

        allocations = pd.DataFrame({

            "MVP": mvp_weights,

            "HRP": hrp_weights

        })

        st.subheader("Allocation Comparison Table")

        allocation_display = allocations.copy()

        allocation_display["MVP"] = (
            allocation_display["MVP"].map(
                lambda x: f"{x:.2%}"
            )
        )

        allocation_display["HRP"] = (
            allocation_display["HRP"].map(
                lambda x: f"{x:.2%}"
            )
        )

        st.dataframe(
            allocation_display,
            use_container_width=True
        )

        # Pie charts
        col1, col2 = st.columns(2)

        with col1:

            fig5 = px.pie(
                names=mvp_weights.index,
                values=mvp_weights.values,
                title="MVP Portfolio Allocation",
                hole=0.25
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )

        with col2:

            fig6 = px.pie(
                names=hrp_weights.index,
                values=hrp_weights.values,
                title="HRP Portfolio Allocation",
                hole=0.25
            )

            st.plotly_chart(
                fig6,
                use_container_width=True
            )

        # ------------------------------------------------
        # 13. RISK-RETURN PROFILE
        # ------------------------------------------------

        st.header("Risk-Return Profile")

        train_portfolios = pd.DataFrame({

            "MVP": train_returns @ mvp_weights,

            "HRP": train_returns @ hrp_weights

        })

        risk_return = pd.DataFrame({

            "Annualized Return": train_portfolios.mean() * 252,

            "Annualized Risk": train_portfolios.std() * np.sqrt(252)

        })

        risk_return["Strategy"] = risk_return.index

        fig7 = px.scatter(
            risk_return,
            x="Annualized Risk",
            y="Annualized Return",
            text="Strategy",
            title="Training Period: Risk vs Return"
        )

        st.plotly_chart(
            fig7,
            use_container_width=True
        )

        # ------------------------------------------------
        # 14. DOWNLOAD RESULTS
        # ------------------------------------------------

        st.header("Download Results")

        st.download_button(
            "Download Portfolio Weights",
            data=allocations.to_csv().encode("utf-8"),
            file_name="portfolio_allocations.csv",
            mime="text/csv"
        )

        results = pd.concat(
            [
                prices.add_prefix("Close_"),
                portfolio_returns.add_prefix("Return_"),
                cumulative_returns.add_prefix("Wealth_")
            ],
            axis=1
        )

        st.download_button(
            "Download Backtesting Results",
            data=results.to_csv().encode("utf-8"),
            file_name="portfolio_backtest.csv",
            mime="text/csv"
        )

        st.caption(
            "MVP: Long-only minimum variance optimization. "
            "HRP: Hierarchical clustering and recursive bisection. "
            "Portfolio weights are estimated on training data and held "
            "constant during testing. Only Close prices are used. "
            "Dividends are excluded."
        )

    except Exception as e:

        st.error(f"An error occurred: {e}")

else:

    st.info(
        "Configure the stocks and dates in the sidebar, "
        "then click Run Portfolio Optimization."
    )