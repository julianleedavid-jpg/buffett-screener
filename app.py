import io
import json
import os
import time
from datetime import datetime
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# --- WATCHLIST FILE SETUP ---
WATCHLIST_FILE = "watchlist.json"

DEFAULT_WATCHLISTS = {
    "Main Watchlist": ["AAPL", "MSFT", "SHEL.L", "LLOY.L", "UKW.L", "BLND.L", "RIO.L"],
}


# --- DYNAMIC FTSE 100 FETCHING ---
@st.cache_data(ttl=86400)
def get_live_ftse100_tickers() -> list:
    """Fetches current FTSE 100 constituents dynamically from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        html_buffer = io.StringIO(response.text)
        tables = pd.read_html(html_buffer)

        for table in tables:
            if "Ticker" in table.columns:
                tickers = (
                    table["Ticker"]
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    + ".L"
                )
                return tickers.tolist()
    except Exception as e:
        st.warning(
            f"Could not fetch live FTSE 100 list dynamically ({e}). Using fallback list."
        )

    return ["SHEL.L", "AZN.L", "HSBA.L", "BP.L", "GSK.L", "LLOY.L", "ULVR.L", "RIO.L"]


# --- DYNAMIC S&P 500 FETCHING ---
@st.cache_data(ttl=86400)
def get_live_sp500_tickers() -> list:
    """Fetches current S&P 500 constituents dynamically from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        html_buffer = io.StringIO(response.text)
        tables = pd.read_html(html_buffer)

        for table in tables:
            if "Symbol" in table.columns:
                tickers = (
                    table["Symbol"]
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                )
                return tickers.tolist()
    except Exception as e:
        st.warning(
            f"Could not fetch live S&P 500 list dynamically ({e}). Using fallback list."
        )

    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BRK-B", "JNJ", "JPM"]


def load_watchlists() -> dict:
    """Loads all watchlists dictionary from local JSON file."""
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {"Main Watchlist": sorted(data)}
                elif isinstance(data, dict) and data:
                    return {k: sorted(v) for k, v in data.items()}
        except Exception:
            return {k: sorted(v) for k, v in DEFAULT_WATCHLISTS.items()}
    return {k: sorted(v) for k, v in DEFAULT_WATCHLISTS.items()}


def save_watchlists(watchlists: dict):
    """Saves watchlists dictionary to local JSON file, keeping ticker lists sorted."""
    sorted_watchlists = {k: sorted(v) for k, v in watchlists.items()}
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(sorted_watchlists, f, indent=4)


# --- RATE-LIMIT RESILIENT DATA FETCHING ---
def fetch_ticker_with_retry(
    ticker_symbol: str, retries: int = 3, delay: float = 1.5
):
    """Fetches a yfinance Ticker object with retries and exponential backoff if rate-limited."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if info:
                return stock, info
        except Exception as e:
            err_msg = str(e).lower()
            if "ratelimit" in err_msg or "429" in err_msg or "too many" in err_msg:
                time.sleep(delay * (attempt + 1))
            else:
                break
    return None, {}


def get_ex_dividend_info(stock: yf.Ticker) -> str:
    """Returns upcoming ex-dividend date if future, checking calendar, info, events, or fallback."""
    today = datetime.now().date()
    candidate_date = None

    try:
        cal = stock.calendar
        if isinstance(cal, dict) and "Ex-Dividend Date" in cal:
            ex_date = cal["Ex-Dividend Date"]
            if ex_date:
                if isinstance(ex_date, (datetime, pd.Timestamp)):
                    candidate_date = ex_date.date()
                elif isinstance(ex_date, str):
                    candidate_date = pd.to_datetime(ex_date).date()
    except Exception:
        pass

    if not candidate_date:
        try:
            ex_ts = stock.info.get("exDividendDate")
            if ex_ts:
                candidate_date = pd.to_datetime(ex_ts, unit="s").date()
        except Exception:
            pass

    if not candidate_date:
        try:
            events = stock.events
            if isinstance(events, pd.DataFrame) and not events.empty:
                col = next(
                    (c for c in ["Ex-Dividend Date", "Dividend Date"] if c in events.columns),
                    None
                )
                if col:
                    future_events = events[events.index.date >= today]
                    if not future_events.empty:
                        candidate_date = future_events.index[0].date()
        except Exception:
            pass

    if candidate_date:
        if candidate_date >= today:
            return candidate_date.strftime("%Y-%m-%d")
        else:
            return f"{candidate_date.strftime('%Y-%m-%d')} (Last Known)"

    try:
        divs = stock.dividends
        if not divs.empty:
            last_date = divs.index[-1].date()
            return f"{last_date.strftime('%Y-%m-%d')} (Last Known)"
    except Exception:
        pass

    return "N/A"


def get_dividend_yield(info: dict) -> float:
    """Extracts annual dividend yield as a percentage, handling yfinance UK decimal/percentage discrepancies."""
    raw_yield = info.get("dividendYield")
    
    div_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    if div_rate and price and price > 0:
        price_in_pounds = price / 100.0 if price > 50 else price
        calculated_yield = (div_rate / price_in_pounds) * 100
        if 0 < calculated_yield < 25:
            return round(calculated_yield, 2)

    if raw_yield is not None:
        if raw_yield < 0.25:
            return round(raw_yield * 100, 2)
        elif raw_yield <= 25.0:
            return round(raw_yield, 2)

    return 0.00


def get_pe_ratio(info: dict) -> float | None:
    """Extracts trailing or forward P/E ratio rounded to 2 decimal places."""
    pe = info.get("trailingPE") or info.get("forwardPE")
    if isinstance(pe, (int, float)) and pe > 0:
        return round(float(pe), 2)
    return None


def get_return_on_equity(info: dict) -> float | None:
    """Extracts Return on Equity as a percentage rounded to 2 decimal places."""
    roe = info.get("returnOnEquity")
    if roe is not None:
        if abs(roe) < 2.0:
            return round(roe * 100, 2)
        return round(roe, 2)
    return None


def get_debt_to_equity(info: dict, stock: yf.Ticker) -> float | None:
    """Extracts Debt to Equity ratio, checking quarterly balance sheet as fallback."""
    dte = info.get("debtToEquity")
    if dte is not None:
        if dte > 10.0:
            return round(dte / 100.0, 2)
        return round(dte, 2)
    
    try:
        bs = stock.quarterly_balance_sheet
        if bs.empty:
            bs = stock.balance_sheet

        if not bs.empty:
            total_debt = None
            for key in ["Total Debt", "Short Long Term Debt", "Long Term Debt"]:
                if key in bs.index:
                    total_debt = bs.loc[key].iloc[0]
                    break
            
            total_equity = None
            for key in ["Total Stockholder Equity", "Stockholders Equity", "Common Stock Equity"]:
                if key in bs.index:
                    total_equity = bs.loc[key].iloc[0]
                    break

            if total_debt is not None and total_equity and total_equity > 0:
                return round(total_debt / total_equity, 2)
    except Exception:
        pass
        
    return None


def get_debt_ratio(info: dict, stock: yf.Ticker) -> float | None:
    """Calculates Debt Ratio (Total Liabilities / Total Assets), utilizing latest quarterly balance sheet."""
    total_liab = info.get("totalLiab") or info.get("totalDebt")
    total_assets = info.get("totalAssets")
    
    if total_liab and total_assets and total_assets > 0:
        return round(total_liab / total_assets, 2)
        
    try:
        bs = stock.quarterly_balance_sheet
        if bs.empty:
            bs = stock.balance_sheet

        if not bs.empty:
            liab_key = next((k for k in ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"] if k in bs.index), None)
            assets_key = next((k for k in ["Total Assets"] if k in bs.index), None)
            
            if liab_key and assets_key:
                latest_liab = bs.loc[liab_key].iloc[0]
                latest_assets = bs.loc[assets_key].iloc[0]
                if latest_liab and latest_assets and latest_assets > 0:
                    return round(latest_liab / latest_assets, 2)
    except Exception:
        pass

    return None


# --- VALUATION ENGINE WITH FULL DUAL-LISTING FALLBACKS ---
def calculate_buffett_value(
    ticker_symbol: str,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    margin_of_safety: float,
    years: int = 10,
):
    """Calculates intrinsic value based on Owner Earnings & DCF with deep fallbacks for dual-listed UK stocks like RIO.L."""
    ticker_symbol = ticker_symbol.strip().upper()
    stock, info = fetch_ticker_with_retry(ticker_symbol)

    if not stock or not info:
        return {
            "Ticker": ticker_symbol,
            "Company": ticker_symbol,
            "Sector": "N/A",
            "Currency": "£" if ticker_symbol.endswith(".L") else "$",
            "Price": None,
            "P/E Ratio": None,
            "ROE (%)": None,
            "Owner Earn/Sh": None,
            "Debt Ratio": None,
            "Debt/Equity": None,
            "Intrinsic Value": None,
            "Buy Price": None,
            "Upside (%)": None,
            "Div Yield (%)": None,
            "Ex-Div Date": "N/A",
            "Status": "ERROR (Rate-Limited/Unavailable)",
        }

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    is_uk_stock = ticker_symbol.endswith(".L")
    currency_symbol = "£" if is_uk_stock else "$"
    financial_currency = info.get("financialCurrency", "")
    sector = info.get("sector", "N/A")

    if current_price and is_uk_stock:
        current_price = current_price / 100.0

    ex_div_date_str = get_ex_dividend_info(stock)
    div_yield_pct = get_dividend_yield(info)
    pe_ratio = get_pe_ratio(info)
    roe = get_return_on_equity(info)

    debt_ratio = get_debt_ratio(info, stock)
    debt_to_equity = get_debt_to_equity(info, stock)

    # --- SHARES OUTSTANDING FALLBACK (Fixes RIO.L missing shares) ---
    shares_outstanding = info.get("sharesOutstanding")
    market_cap = info.get("marketCap")
    
    if not shares_outstanding and market_cap and current_price and current_price > 0:
        shares_outstanding = market_cap / current_price

    if not current_price or not shares_outstanding:
        return {
            "Ticker": ticker_symbol,
            "Company": info.get("shortName", ticker_symbol),
            "Sector": sector,
            "Currency": currency_symbol,
            "Price": current_price,
            "P/E Ratio": pe_ratio,
            "ROE (%)": roe,
            "Owner Earn/Sh": None,
            "Debt Ratio": debt_ratio,
            "Debt/Equity": debt_to_equity,
            "Intrinsic Value": None,
            "Buy Price": None,
            "Upside (%)": None,
            "Div Yield (%)": div_yield_pct,
            "Ex-Div Date": ex_div_date_str,
            "Status": "ERROR (Missing Price/Shares)",
        }

    # --- CASH FLOW & OWNER EARNINGS EXTRACTION WITH EXPANDED FALLBACKS ---
    operating_cf = None
    capex = 0.0

    cashflow = stock.cashflow
    if not cashflow.empty:
        latest_cf = cashflow.iloc[:, 0]
        operating_cf = latest_cf.get("Operating Cash Flow") or latest_cf.get("Free Cash Flow")
        raw_capex = latest_cf.get("Capital Expenditure") or latest_cf.get("Capital Expenditures")
        if raw_capex is not None:
            capex = abs(raw_capex)

    # Fallback 1: Direct Free Cash Flow from info dict
    if operating_cf is None:
        operating_cf = info.get("freeCashflow") or info.get("operatingCashflow")

    # Fallback 2: Comprehensive Net Income Search from Financials
    if operating_cf is None:
        try:
            financials = stock.financials
            if not financials.empty:
                latest_fin = financials.iloc[:, 0]
                for net_inc_key in [
                    "Net Income",
                    "Net Income Common Stockholders",
                    "Net Income From Continuing Operation",
                    "Net Income Including Noncontrolling Interests",
                    "Normalized Income"
                ]:
                    if net_inc_key in latest_fin.index:
                        operating_cf = latest_fin.get(net_inc_key)
                        break
        except Exception:
            pass

    # Fallback 3: EPS-based Owner Earnings (Trailing or Forward EPS)
    eps = info.get("trailingEps") or info.get("forwardEps")
    if operating_cf is None and eps:
        operating_cf = eps * shares_outstanding

    if operating_cf is None:
        return {
            "Ticker": ticker_symbol,
            "Company": info.get("shortName", ticker_symbol),
            "Sector": sector,
            "Currency": currency_symbol,
            "Price": current_price,
            "P/E Ratio": pe_ratio,
            "ROE (%)": roe,
            "Owner Earn/Sh": None,
            "Debt Ratio": debt_ratio,
            "Debt/Equity": debt_to_equity,
            "Intrinsic Value": None,
            "Buy Price": None,
            "Upside (%)": None,
            "Div Yield (%)": div_yield_pct,
            "Ex-Div Date": ex_div_date_str,
            "Status": "N/A (Data Unavailable)",
        }

    owner_earnings = operating_cf - capex
    owner_earnings_per_share = owner_earnings / shares_outstanding

    if is_uk_stock and financial_currency.upper() in ["GBX", "GBP"]:
        if financial_currency.upper() == "GBX":
            owner_earnings_per_share = owner_earnings_per_share / 100.0

    pv_future_cash_flows = 0.0
    projected_oe = owner_earnings_per_share

    for year in range(1, years + 1):
        projected_oe *= 1 + growth_rate
        discounted_value = projected_oe / ((1 + discount_rate) ** year)
        pv_future_cash_flows += discounted_value

    terminal_value = (
        projected_oe
        * (1 + terminal_growth_rate)
        / (discount_rate - terminal_growth_rate)
    )
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** years)

    intrinsic_value = pv_future_cash_flows + discounted_terminal_value
    buy_price = intrinsic_value * (1 - margin_of_safety)
    upside = ((intrinsic_value - current_price) / current_price) * 100

    partial_margin_price = intrinsic_value * 0.90

    if current_price <= buy_price:
        status = "DEEP VALUE (Very Undervalued)"
    elif current_price <= partial_margin_price:
        status = "BUY (Moderate Margin)"
    elif current_price <= intrinsic_value:
        status = "FAIR (No Margin)"
    else:
        status = "OVERVALUED"

    return {
        "Ticker": ticker_symbol,
        "Company": info.get("shortName", ticker_symbol),
        "Sector": sector,
        "Currency": currency_symbol,
        "Price": current_price,
        "P/E Ratio": pe_ratio,
        "ROE (%)": roe,
        "Owner Earn/Sh": owner_earnings_per_share,
        "Debt Ratio": debt_ratio,
        "Debt/Equity": debt_to_equity,
        "Intrinsic Value": intrinsic_value,
        "Buy Price": buy_price,
        "Upside (%)": upside,
        "Div Yield (%)": div_yield_pct,
        "Ex-Div Date": ex_div_date_str,
        "Status": status,
    }


def render_valuation_table(df):
    def color_status(val):
        str_val = str(val)
        if "DEEP VALUE" in str_val:
            return "background-color: #c3e6cb; color: #0b5121; font-weight: bold;"
        elif "BUY" in str_val:
            return "background-color: #d4edda; color: #155724;"
        elif "FAIR" in str_val:
            return "background-color: #fff3cd; color: #856404;"
        elif "OVERVALUED" in str_val:
            return "background-color: #f8d7da; color: #721c24;"
        return ""

    format_dict = {
        "Price": "{:.2f}",
        "P/E Ratio": "{:.2f}",
        "ROE (%)": "{:.2f}",
        "Owner Earn/Sh": "{:.2f}",
        "Debt Ratio": "{:.2f}",
        "Debt/Equity": "{:.2f}",
        "Intrinsic Value": "{:.2f}",
        "Buy Price": "{:.2f}",
        "Upside (%)": "{:.2f}",
        "Div Yield (%)": "{:.2f}",
    }

    styled_df = df.style.format(format_dict, na_rep="N/A").map(color_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True, height=750)


# --- STREAMLIT UI SETUP ---
st.set_page_config(
    page_title="Buffett Value Calculator", page_icon="📈", layout="wide"
)

# --- INJECT CUSTOM CSS FOR FLOATING / STICKY TABLE HEADERS ---
st.markdown(
    """
    <style>
    /* Pin and float DataFrame headers */
    div[data-testid="stDataFrame"] div[role="columnheader"],
    div[data-testid="stDataFrame"] [data-testid="stTable"] th {
        position: sticky !important;
        top: 0 !important;
        z-index: 100 !important;
        background-color: #1e293b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_watchlists()

if "active_watchlist_name" not in st.session_state:
    st.session_state.active_watchlist_name = list(st.session_state.watchlists.keys())[0]

if "single_val_result" not in st.session_state:
    st.session_state.single_val_result = None

if "watchlist_results_df" not in st.session_state:
    st.session_state.watchlist_results_df = None

if "ftse_results_df" not in st.session_state:
    st.session_state.ftse_results_df = None

if "sp500_results_df" not in st.session_state:
    st.session_state.sp500_results_df = None


st.title("📈 Buffett Value Investing & Portfolio Screener")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Valuation Parameters")
growth_rate = (
    st.sidebar.slider(
        "Expected 10Y Growth Rate (%)", 0.0, 30.0, 8.0, step=0.5
    )
    / 100.0
)
discount_rate = (
    st.sidebar.slider(
        "Target Discount Rate (%)", 5.0, 20.0, 10.0, step=0.5
    )
    / 100.0
)
terminal_rate = (
    st.sidebar.slider(
        "Terminal Growth Rate (%)", 0.0, 5.0, 2.5, step=0.25
    )
    / 100.0
)
margin_of_safety = (
    st.sidebar.slider("Margin of Safety (%)", 0, 50, 25, step=5) / 100.0
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Single Stock Analysis",
        "📋 Personal Watchlist Screener",
        "🇬🇧 Live FTSE 100 Screener",
        "🇺🇸 Live S&P 500 Screener",
    ]
)

# ==========================================
# TAB 1: SINGLE STOCK ANALYSIS
# ==========================================
with tab1:
    st.subheader("Analyze an Individual Stock")
    single_ticker = (
        st.text_input(
            "Enter Ticker Symbol (e.g., AAPL, RIO.L, UKW.L, BLND.L)", value="RIO.L"
        )
        .strip()
        .upper()
    )

    if st.button("Calculate Single Ticker", type="primary"):
        with st.spinner(f"Evaluating {single_ticker}..."):
            st.session_state.single_val_result = calculate_buffett_value(
                ticker_symbol=single_ticker,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_rate,
                margin_of_safety=margin_of_safety,
            )

    if st.session_state.single_val_result:
        val = st.session_state.single_val_result
        curr = val["Currency"]
        st.subheader(f"{val['Company']} ({val['Ticker']}) — Sector: {val['Sector']}")

        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
        col1.metric(
            "Current Price",
            f"{curr}{val['Price']:.2f}" if val["Price"] is not None else "N/A",
        )
        col2.metric("P/E Ratio", f"{val['P/E Ratio']:.2f}" if isinstance(val["P/E Ratio"], (int, float)) else "N/A")
        col3.metric("ROE (%)", f"{val['ROE (%)']:.2f}%" if isinstance(val["ROE (%)"], (int, float)) else "N/A")
        col4.metric(
            "Owner Earn / Share",
            f"{curr}{val['Owner Earn/Sh']:.2f}"
            if val["Owner Earn/Sh"] is not None
            else "N/A",
        )
        col5.metric("Debt Ratio", f"{val['Debt Ratio']:.2f}" if isinstance(val["Debt Ratio"], (int, float)) else "N/A")
        col6.metric("Debt/Equity", f"{val['Debt/Equity']:.2f}" if isinstance(val["Debt/Equity"], (int, float)) else "N/A")
        col7.metric(
            "Intrinsic Value",
            f"{curr}{val['Intrinsic Value']:.2f}"
            if val["Intrinsic Value"] is not None
            else "N/A",
            delta=f"{val['Upside (%)']:.2f}% Upside"
            if val["Upside (%)"] is not None
            else None,
        )
        col8.metric(
            f"Buy Price ({margin_of_safety * 100:.0f}% MOS)",
            f"{curr}{val['Buy Price']:.2f}" if val["Buy Price"] is not None else "N/A",
        )
        col9.metric("Annual Div Yield", f"{val['Div Yield (%)']:.2f}%" if isinstance(val["Div Yield (%)"], (int, float)) else "N/A")
        col10.metric("Ex-Dividend Date", val["Ex-Div Date"])

        st.divider()

        if "DEEP VALUE" in val["Status"]:
            st.balloons()
            st.success(
                f"**Strong Buy / Deep Value Signal:** Price ({curr}{val['Price']:.2f}) is below target entry price ({curr}{val['Buy Price']:.2f}) with full margin of safety."
            )
        elif "BUY" in val["Status"]:
            st.success(
                f"**Buy Signal:** Price ({curr}{val['Price']:.2f}) is below intrinsic value ({curr}{val['Intrinsic Value']:.2f}) with moderate margin of safety."
            )
        elif "FAIR" in val["Status"]:
            st.warning(
                f"**Fairly Valued:** Price ({curr}{val['Price']:.2f}) is below intrinsic value ({curr}{val['Intrinsic Value']:.2f}), but lacks desired margin of safety."
            )
        elif "OVERVALUED" in val["Status"]:
            st.error(
                f"**Overvalued:** Price ({curr}{val['Price']:.2f}) exceeds intrinsic value ({curr}{val['Intrinsic Value']:.2f})."
            )
        else:
            st.info(f"**Valuation Status:** {val['Status']}")

# ==========================================
# TAB 2: PERSONAL WATCHLIST SCREENER
# ==========================================
with tab2:
    st.subheader("Manage & Run Personal Watchlists")

    col_sel, col_new, col_del_wl = st.columns([2, 2, 1])

    with col_sel:
        all_wl_names = list(st.session_state.watchlists.keys())
        selected_wl = st.selectbox(
            "Select Active Watchlist",
            all_wl_names,
            index=all_wl_names.index(st.session_state.active_watchlist_name)
            if st.session_state.active_watchlist_name in all_wl_names
            else 0,
        )
        st.session_state.active_watchlist_name = selected_wl

    with col_new:
        new_wl_name = st.text_input("Create New Watchlist Name").strip()
        if st.button("📁 Create Watchlist"):
            if new_wl_name and new_wl_name not in st.session_state.watchlists:
                st.session_state.watchlists[new_wl_name] = []
                st.session_state.active_watchlist_name = new_wl_name
                save_watchlists(st.session_state.watchlists)
                st.toast(f"Created Watchlist '{new_wl_name}'!", icon="✅")
                st.rerun()

    with col_del_wl:
        st.write("")
        st.write("")
        if st.button("🗑️ Delete Current Watchlist"):
            if len(st.session_state.watchlists) <= 1:
                st.error("Cannot delete your only watchlist.")
            else:
                del_name = st.session_state.active_watchlist_name
                del st.session_state.watchlists[del_name]
                st.session_state.active_watchlist_name = list(st.session_state.watchlists.keys())[0]
                save_watchlists(st.session_state.watchlists)
                st.toast(f"Deleted Watchlist '{del_name}'.", icon="🗑️")
                st.rerun()

    active_list = sorted(st.session_state.watchlists[st.session_state.active_watchlist_name])

    st.divider()

    col_add, col_del = st.columns(2)

    with col_add:
        new_ticker = (
            st.text_input("Add Ticker to Active Watchlist (e.g., BP.L, MSFT)")
            .strip()
            .upper()
        )
        if st.button("➕ Add Ticker"):
            if new_ticker and new_ticker not in st.session_state.watchlists[st.session_state.active_watchlist_name]:
                st.session_state.watchlists[st.session_state.active_watchlist_name].append(new_ticker)
                save_watchlists(st.session_state.watchlists)
                st.toast(f"Added {new_ticker} to {st.session_state.active_watchlist_name}!", icon="✅")
                st.rerun()

    with col_del:
        if active_list:
            remove_ticker = st.selectbox("Remove Ticker from Active Watchlist", active_list)
            if st.button("🗑️ Remove Ticker"):
                st.session_state.watchlists[st.session_state.active_watchlist_name].remove(remove_ticker)
                save_watchlists(st.session_state.watchlists)
                st.toast(f"Removed {remove_ticker}.", icon="🗑️")
                st.rerun()

    st.write(
        f"**Active Watchlist: `{st.session_state.active_watchlist_name}` ({len(active_list)} stocks):** "
        + (", ".join(active_list) if active_list else "*List is currently empty*")
    )

    st.divider()

    if st.button("🚀 Run Active Watchlist Batch", type="primary"):
        if not active_list:
            st.warning("Your active watchlist is empty. Add some tickers first.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_tickers = len(active_list)

            for idx, symbol in enumerate(active_list):
                status_text.text(
                    f"Evaluating {symbol} ({idx + 1}/{total_tickers})..."
                )
                res = calculate_buffett_value(
                    ticker_symbol=symbol,
                    growth_rate=growth_rate,
                    discount_rate=discount_rate,
                    terminal_growth_rate=terminal_rate,
                    margin_of_safety=margin_of_safety,
                )
                if res:
                    results.append(res)

                time.sleep(0.2)
                progress_bar.progress((idx + 1) / total_tickers)

            status_text.success(f"Valuation Complete for '{st.session_state.active_watchlist_name}'!")
            st.session_state.watchlist_results_df = pd.DataFrame(results)

    if st.session_state.watchlist_results_df is not None:
        df_wl = st.session_state.watchlist_results_df
        render_valuation_table(df_wl)

        csv_data = df_wl.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"{st.session_state.active_watchlist_name}_buffett_valuation.csv",
            mime="text/csv",
        )

# ==========================================
# TAB 3: LIVE FTSE 100 SCREENER
# ==========================================
with tab3:
    st.subheader("🇬🇧 Live FTSE 100 Index Screener")
    st.markdown(
        "Fetches the **live FTSE 100 index constituents** dynamically and runs a full Warren Buffett valuation across all members."
    )

    with st.spinner("Fetching live FTSE 100 constituent list..."):
        ftse_live_tickers = get_live_ftse100_tickers()

    st.info(
        f"**Active FTSE 100 Members Loaded:** {len(ftse_live_tickers)} stocks ready to scan."
    )

    if st.button("🚀 Run Live FTSE 100 Scan", type="primary"):
        results_ftse = []
        progress_bar_ftse = st.progress(0)
        status_text_ftse = st.empty()

        total_ftse = len(ftse_live_tickers)

        for idx, symbol in enumerate(ftse_live_tickers):
            status_text_ftse.text(
                f"Scanning FTSE 100 ({idx + 1}/{total_ftse}): {symbol}..."
            )
            res = calculate_buffett_value(
                ticker_symbol=symbol,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_rate,
                margin_of_safety=margin_of_safety,
            )
            if res:
                results_ftse.append(res)

            time.sleep(0.2)
            progress_bar_ftse.progress((idx + 1) / total_ftse)

        status_text_ftse.success("FTSE 100 Live Scan Complete!")
        st.session_state.ftse_results_df = pd.DataFrame(results_ftse)

    if st.session_state.ftse_results_df is not None:
        df_ftse = st.session_state.ftse_results_df
        render_valuation_table(df_ftse)

        csv_ftse_data = df_ftse.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download FTSE 100 Scan as CSV",
            data=csv_ftse_data,
            file_name="ftse100_live_buffett_scan.csv",
            mime="text/csv",
        )

# ==========================================
# TAB 4: LIVE S&P 500 SCREENER
# ==========================================
with tab4:
    st.subheader("🇺🇸 Live S&P 500 Index Screener")
    st.markdown(
        "Fetches the **live S&P 500 index constituents** dynamically and runs a full Warren Buffett valuation across all members."
    )

    with st.spinner("Fetching live S&P 500 constituent list..."):
        sp500_live_tickers = get_live_sp500_tickers()

    st.info(
        f"**Active S&P 500 Members Loaded:** {len(sp500_live_tickers)} stocks ready to scan."
    )

    if st.button("🚀 Run Live S&P 500 Scan", type="primary"):
        results_sp500 = []
        progress_bar_sp500 = st.progress(0)
        status_text_sp500 = st.empty()

        total_sp500 = len(sp500_live_tickers)

        for idx, symbol in enumerate(sp500_live_tickers):
            status_text_sp500.text(
                f"Scanning S&P 500 ({idx + 1}/{total_sp500}): {symbol}..."
            )
            res = calculate_buffett_value(
                ticker_symbol=symbol,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_rate,
                margin_of_safety=margin_of_safety,
            )
            if res:
                results_sp500.append(res)

            time.sleep(0.2)
            progress_bar_sp500.progress((idx + 1) / total_sp500)

        status_text_sp500.success("S&P 500 Live Scan Complete!")
        st.session_state.sp500_results_df = pd.DataFrame(results_sp500)

    if st.session_state.sp500_results_df is not None:
        df_sp500 = st.session_state.sp500_results_df
        render_valuation_table(df_sp500)

        csv_sp500_data = df_sp500.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download S&P 500 Scan as CSV",
            data=csv_sp500_data,
            file_name="sp500_live_buffett_scan.csv",
            mime="text/csv",
        )
