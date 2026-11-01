"""Stock Forecast – web app.

Run from the repository root:

    poetry run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stock_forecast.data import (
    TICKER_CATALOG,
    VALID_INTERVALS,
    VALID_PERIODS,
    download_series,
    get_ticker_info,
    is_valid_isin,
    lookup_isin,
)
from stock_forecast.ensemble import ENSEMBLE_METHODS, EnsembleResult
from stock_forecast.metrics import SELECTION_METRICS
from stock_forecast.models import FAMILIES, MODEL_REGISTRY, Param
from stock_forecast.pipeline import run_pipeline, select_best_model
from stock_forecast.plot import plot_backtest, plot_forecast

st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide")

CUSTOM_TICKER = "✏️ Ticker personalizzato"
ISIN_LOOKUP = "🔎 Codice ISIN"
DEFAULT_MODELS = ("Chronos-2", "Timestral", "TimesFM-3", "TiRex", "Chronos-Bolt", "Theta", "Naive")
STRATEGY_LABELS = {
    "auto": "🤖 Automatica – ensemble solo se batte il miglior modello",
    "best": "🏆 Miglior modello nel backtest",
    "ensemble": "🧩 Ensemble dei migliori modelli",
    "single": "🎯 Un modello a scelta",
}
METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)", "rel_mae": "MAE / Naive"}
METHOD_LABELS = {"weighted": "Media pesata (inverso dell'errore)", "mean": "Media semplice", "median": "Mediana"}
PERIOD_LABELS = {"6mo": "6 mesi", "1y": "1 anno", "2y": "2 anni", "5y": "5 anni", "10y": "10 anni", "max": "Tutto"}
INTERVAL_LABELS = {"1d": "Giornaliera", "1wk": "Settimanale", "1mo": "Mensile"}
HISTORY_LABELS = {90: "3 mesi", 180: "6 mesi", 365: "1 anno", 730: "2 anni", None: "Tutto"}


@st.cache_data(ttl=3600, show_spinner="Scarico i dati da Yahoo Finance…")
def load_series(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return download_series(ticker, period=period, interval=interval)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_info(ticker: str) -> dict:
    return get_ticker_info(ticker)


@st.cache_data(ttl=24 * 3600, show_spinner="Cerco le quotazioni dell'ISIN…")
def find_listings(isin: str) -> list[dict[str, str]]:
    listings = lookup_isin(isin)
    if not listings:
        # Raising instead of returning [] keeps a network hiccup out of the cache.
        msg = "Nessuna quotazione trovata su Yahoo Finance: obbligazioni e fondi non quotati non sono coperti."
        raise LookupError(msg)
    return listings


def param_widget(model: str, param: Param) -> object:
    """Render the widget of one model parameter and return its value."""
    key = f"param_{model}_{param.name}"
    help_text = param.help or None
    if param.choices is not None:
        return st.selectbox(param.label, param.choices, index=param.choices.index(param.default), key=key, help=help_text)

    cast = int if isinstance(param.default, int) else float
    return cast(
        st.number_input(
            param.label,
            min_value=None if param.min_value is None else cast(param.min_value),
            max_value=None if param.max_value is None else cast(param.max_value),
            value=param.default,
            step=None if param.step is None else cast(param.step),
            key=key,
            help=help_text,
        )
    )


def metrics_table(validation_results: dict[str, dict], metric: str) -> pd.DataFrame:
    """Backtest metrics of every model, best first."""
    rows = []
    for name, data in validation_results.items():
        m = data["metrics"]
        rows.append(
            {
                "Modello": name,
                "Famiglia": MODEL_REGISTRY[name].family if name in MODEL_REGISTRY else "Ensemble",
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "MAPE (%)": m["mape"],
                "MAE / Naive": m.get("rel_mae"),
                "Copertura 80% (%)": m.get("coverage"),
                "Errore": data.get("error") or "",
            }
        )
    table = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return table.sort_values(METRIC_LABELS[metric], na_position="last")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – asset & horizon
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Stock Forecast")

    category = st.selectbox("Categoria", [*TICKER_CATALOG, ISIN_LOOKUP, CUSTOM_TICKER])
    if category == CUSTOM_TICKER:
        ticker = st.text_input("Ticker Yahoo Finance", value="AAPL", help="Es. AAPL, ENI.MI, GC=F, BTC-USD").strip().upper()
        asset_name = ticker
    elif category == ISIN_LOOKUP:
        isin = st.text_input(
            "Codice ISIN",
            placeholder="es. IE00B4L5Y983",
            help="Azioni, ETF ed ETC quotati, risolti con Yahoo Finance e OpenFIGI. "
            "Obbligazioni e fondi non quotati non sono coperti.",
        ).strip()
        listings = []
        if isin and not is_valid_isin(isin):
            st.error("Codice ISIN non valido: controlla i 12 caratteri.")
        elif isin:
            try:
                listings = find_listings(isin.upper())
            except LookupError as e:
                st.warning(str(e))
        if listings:
            by_symbol = {listing["symbol"]: listing for listing in listings}
            ticker = st.selectbox(
                "Quotazione",
                list(by_symbol),
                format_func=lambda symbol: f"{symbol} · {by_symbol[symbol]['exchange']}",
                help="Lo stesso strumento su borse diverse: le quotazioni di Milano sono in EUR.",
            )
            asset_name = by_symbol[ticker]["name"]
        else:
            ticker = asset_name = ""
    else:
        assets = TICKER_CATALOG[category]
        asset_name = st.selectbox("Asset", list(assets), format_func=lambda name: f"{name} · {assets[name]}")
        ticker = assets[asset_name]

    period = st.selectbox("Storico", VALID_PERIODS, index=VALID_PERIODS.index("5y"), format_func=PERIOD_LABELS.get)
    interval = st.selectbox("Frequenza", VALID_INTERVALS, format_func=INTERVAL_LABELS.get)
    horizon = st.slider("Orizzonte di previsione (barre)", min_value=1, max_value=120, value=15)

    run_clicked = st.button("🚀 Avvia previsione", type="primary", width="stretch", disabled=not ticker)
    st.caption("Previsioni statistiche a scopo didattico: non sono consigli di investimento.")

st.header(f"{asset_name} ({ticker})" if asset_name != ticker else (ticker or "Stock Forecast"))
run_area = st.container()
tab_forecast, tab_backtest, tab_settings, tab_data = st.tabs(["📈 Previsione", "🧪 Backtest", "⚙️ Impostazioni", "🗃️ Dati"])

# ──────────────────────────────────────────────────────────────────────────────
# Settings tab – strategy, models, parameters
# ──────────────────────────────────────────────────────────────────────────────

with tab_settings:
    available = [name for name, spec in MODEL_REGISTRY.items() if spec.available]
    missing = [spec for spec in MODEL_REGISTRY.values() if not spec.available]

    st.subheader("Strategia")
    strategy = st.radio(
        "Come scegliere la previsione finale",
        list(STRATEGY_LABELS),
        format_func=STRATEGY_LABELS.get,
        label_visibility="collapsed",
    )

    st.subheader("Modelli")
    if strategy == "single":
        single = st.selectbox(
            "Modello",
            available,
            index=available.index("Chronos-2"),
            format_func=lambda name: f"{name}  ·  {MODEL_REGISTRY[name].family}",
        )
        st.caption(MODEL_REGISTRY[single].description)
        models = [single]
    else:
        models = []
        columns = st.columns(2)
        families = [family for family in FAMILIES if any(MODEL_REGISTRY[n].family == family for n in available)]
        for i, family in enumerate(families):
            names = [name for name in available if MODEL_REGISTRY[name].family == family]
            with columns[i % 2]:
                models += st.multiselect(
                    family,
                    names,
                    default=[name for name in names if name in DEFAULT_MODELS],
                    key=f"models_{family}",
                    placeholder="Nessun modello",
                    help="\n\n".join(f"**{name}** – {MODEL_REGISTRY[name].description}" for name in names),
                )
    if missing:
        st.caption(
            "Non installati: " + ", ".join(f"**{spec.name}** (`poetry install --extras {spec.extra}`)" for spec in missing)
        )
    if any(MODEL_REGISTRY[name].slow for name in models):
        st.warning(
            "N-HiTS e N-BEATS vengono addestrati da zero per ogni finestra di backtest: possono servire diversi minuti.",
            icon="⏳",
        )

    model_params: dict[str, dict] = {}
    tunable = [name for name in models if MODEL_REGISTRY[name].params]
    if tunable:
        st.subheader("Parametri dei modelli")
        columns = st.columns(2)
        for i, name in enumerate(tunable):
            with columns[i % 2], st.expander(name):
                model_params[name] = {p.name: param_widget(name, p) for p in MODEL_REGISTRY[name].params}

    st.subheader("Validazione")
    col_windows, col_metric = st.columns(2)
    n_windows = col_windows.slider(
        "Finestre di backtest",
        min_value=1,
        max_value=6,
        value=3,
        help="Ogni modello viene valutato su N blocchi consecutivi delle ultime osservazioni "
        "(ciascuno lungo quanto l'orizzonte). Più finestre = selezione più affidabile ma più lenta.",
    )
    metric = col_metric.selectbox(
        "Metrica di selezione",
        SELECTION_METRICS,
        format_func=METRIC_LABELS.get,
        help="MAE / Naive < 1 significa che il modello batte il random walk sulle stesse finestre.",
    )

    ensemble_method, top_k = "weighted", 3
    if strategy in ("auto", "ensemble"):
        st.subheader("Ensemble")
        col_method, col_k = st.columns(2)
        ensemble_method = col_method.selectbox("Combinazione", ENSEMBLE_METHODS, format_func=METHOD_LABELS.get)
        top_k = col_k.slider("Modelli nell'ensemble", min_value=2, max_value=6, value=3)

    with st.expander("ℹ️ Tutti i modelli"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Modello": spec.name,
                        "Famiglia": spec.family,
                        "Descrizione": spec.description,
                        "Installato": "✅" if spec.available else f"❌ extra '{spec.extra}'",
                    }
                    for spec in MODEL_REGISTRY.values()
                ]
            ),
            hide_index=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────


def execute_run(pipeline_kwargs: dict) -> dict | None:
    """Download the series and run the pipeline, reporting progress; ``None`` on failure."""
    if not pipeline_kwargs["models"]:
        st.error("Seleziona almeno un modello nella tab ⚙️ Impostazioni.")
        return None
    try:
        df = load_series(ticker, period, interval)
    except Exception as e:
        st.error(f"Impossibile scaricare **{ticker}**: {e}")
        return None

    with st.status(f"Previsione di {asset_name} in corso…", expanded=True) as status:
        bar = st.progress(0.0, text="Avvio…")

        def on_progress(step: int, total: int, message: str) -> None:
            bar.progress(min(step / max(total, 1), 1.0), text=message)

        try:
            val_results, final = run_pipeline(df, horizon, progress=on_progress, **pipeline_kwargs)
        except Exception as e:
            status.update(label="Previsione non riuscita", state="error")
            st.error(str(e))
            return None
        status.update(label=f"Previsione completata con {final.model_name}", state="complete", expanded=False)

    return {
        "ticker": ticker,
        "name": asset_name,
        "df": df,
        "info": load_info(ticker),
        "horizon": horizon,
        "metric": pipeline_kwargs["selection_metric"],
        "val_results": val_results,
        "final": final,
    }


if run_clicked:
    with run_area:
        result = execute_run(
            {
                "models": models,
                "selection_metric": metric,
                "n_windows": n_windows,
                "model_params": model_params,
                "strategy": "best" if strategy == "single" else strategy,
                "ensemble_method": ensemble_method,
                "top_k": top_k,
            }
        )
    if result is not None:
        st.session_state["run"] = result

run = st.session_state.get("run")
if run is not None and run["ticker"] != ticker:
    with run_area:
        st.info(f"I risultati mostrati sono di **{run['name']} ({run['ticker']})**. Premi *Avvia previsione* per {ticker}.")

# ──────────────────────────────────────────────────────────────────────────────
# Forecast tab
# ──────────────────────────────────────────────────────────────────────────────

with tab_forecast:
    if run is None:
        st.info("Scegli un asset nella barra laterale, regola le ⚙️ Impostazioni e premi **🚀 Avvia previsione**.")
    else:
        df, final = run["df"], run["final"]
        fc = final.forecast
        last_price, prev_price = float(df["y"].iloc[-1]), float(df["y"].iloc[-2])
        decimals = 2 if abs(last_price) >= 10 else 4  # FX rates and yields need more precision

        def price(value: float) -> str:
            return f"{value:,.{decimals}f}"

        def change(value: float) -> str:
            return f"{value / last_price - 1:+.2%}"

        end_date = f"{fc['ds'].iloc[-1]:%d/%m/%Y}"
        end, low, high = (float(fc[col].iloc[-1]) for col in ("yhat", "yhat_lo", "yhat_hi"))
        col_last, col_end, col_low, col_high = st.columns(4)
        col_last.metric(
            "Ultimo prezzo",
            price(last_price),
            f"{last_price / prev_price - 1:+.2%}",
            help=f"Chiusura del {df['ds'].iloc[-1]:%d/%m/%Y}",
        )
        col_end.metric(
            f"Previsione a {run['horizon']} barre", price(end), change(end), help=f"Mediana prevista per il {end_date}"
        )
        col_low.metric("Minimo (intervallo 80%)", price(low), change(low), delta_color="off", help=f"Al {end_date}")
        col_high.metric("Massimo (intervallo 80%)", price(high), change(high), delta_color="off", help=f"Al {end_date}")

        details = [f"Modello: **{final.model_name}**"]
        if isinstance(final, EnsembleResult):
            details.append(" · ".join(f"{name} {weight:.0%}" for name, weight in final.weights.items()))
        rel_mae = final.metrics.get("rel_mae")
        if rel_mae is not None and np.isfinite(rel_mae):
            verdict = "batte" if rel_mae < 1 else "non batte"
            details.append(f"MAE / Naive nel backtest **{rel_mae:.3f}**: {verdict} il random walk («domani = oggi»)")
        st.caption("  \n".join(details))

        col_history, col_members = st.columns([3, 1])
        history_days = col_history.select_slider(
            "Storico nel grafico",
            options=list(HISTORY_LABELS),
            value=180,
            format_func=HISTORY_LABELS.get,
        )
        show_members = isinstance(final, EnsembleResult) and col_members.toggle("Mostra i modelli dell'ensemble", value=True)

        st.plotly_chart(
            plot_forecast(final, ticker=run["name"], last_n_days=history_days, show_members=show_members),
            theme=None,
        )

        table = fc.rename(columns={"ds": "Data", "yhat": "Previsione", "yhat_lo": "Minimo 80%", "yhat_hi": "Massimo 80%"})
        price_column = st.column_config.NumberColumn(format=f"%.{decimals}f")
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "Data": st.column_config.DateColumn(format="DD/MM/YYYY"),
                "Previsione": price_column,
                "Minimo 80%": price_column,
                "Massimo 80%": price_column,
            },
        )
        st.download_button(
            "⬇️ Scarica la previsione (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            file_name=f"{run['ticker'].replace('=', '_').replace('^', '')}_forecast.csv",
            mime="text/csv",
        )

# ──────────────────────────────────────────────────────────────────────────────
# Backtest tab
# ──────────────────────────────────────────────────────────────────────────────

with tab_backtest:
    if run is None:
        st.info("I risultati del backtest compariranno qui dopo la prima previsione.")
    else:
        val_results = run["val_results"]
        best = select_best_model(val_results, run["metric"])
        st.markdown(f"Migliore per **{METRIC_LABELS[run['metric']]}**: **{best}**")

        number = st.column_config.NumberColumn(format="%.3f")
        st.dataframe(
            metrics_table(val_results, run["metric"]),
            hide_index=True,
            column_config={
                "MAE": number,
                "RMSE": number,
                "MAPE (%)": st.column_config.NumberColumn(format="%.2f"),
                "MAE / Naive": number,
                "Copertura 80% (%)": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.caption(
            "**MAE / Naive** < 1 → batte il random walk. "
            "**Copertura** vicina all'80% → intervalli di previsione ben calibrati."
        )
        st.plotly_chart(plot_backtest(val_results, run["df"]), theme=None)

# ──────────────────────────────────────────────────────────────────────────────
# Data tab
# ──────────────────────────────────────────────────────────────────────────────

with tab_data:
    if run is None:
        st.info("I dati scaricati compariranno qui dopo la prima previsione.")
    else:
        info, df = run["info"], run["df"]
        col_name, col_currency, col_exchange, col_rows = st.columns(4)
        col_name.metric("Nome", info.get("name", run["ticker"]))
        col_currency.metric("Valuta", info.get("currency", "—"))
        col_exchange.metric("Mercato", info.get("exchange", "—"))
        col_rows.metric("Osservazioni", f"{len(df):,}")
        st.dataframe(df.drop(columns="y").iloc[::-1], hide_index=True)
        st.download_button(
            "⬇️ Scarica i dati (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{run['ticker'].replace('=', '_').replace('^', '')}_data.csv",
            mime="text/csv",
        )
