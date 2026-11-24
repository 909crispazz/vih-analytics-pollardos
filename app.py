import streamlit as st

from config.settings import PALETTE
from data.api_client import fetch_epidemiology
from data.dataset import generate_clinical_dataset
from ml.pipeline import train_models
from ui.components import (
    load_stylesheet,
    render_footer,
    render_hero,
    render_kpi_card,
    render_sidebar,
)
from ui.views import (
    render_epidemiology_tab,
    render_kmeans_tab,
    render_random_forest_tab,
    render_simulator_tab,
    render_stanford_tab,
)

st.set_page_config(
    page_title="VIH-Analytics Studio | Equipo Pollardos",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_dataset():
    return generate_clinical_dataset(seed=42)


@st.cache_resource
def get_models(df_base, k_val, trees):
    return train_models(df_base, n_clusters=k_val, n_estimators=trees)


@st.cache_data(ttl=3600)
def get_epidemiology(use_live):
    return fetch_epidemiology(country="MEX", use_live_api=use_live)


def main():
    load_stylesheet()
    k_val, trees, live_api = render_sidebar()

    df_base = get_dataset()
    bundle = get_models(df_base, k_val, trees)
    df_epi, epi_source = get_epidemiology(live_api)

    render_hero(bundle["engine"])

    df_mod = bundle["df"]
    total = len(df_mod)
    failures = int(df_mod["fallo_virologico"].sum())
    fail_pct = failures / total * 100.0
    f1_fail = bundle["report"]["Fallo Virologico (1)"]["f1-score"] * 100.0
    last_art = df_epi["Cobertura_TAR_Pct"].iloc[-1]
    last_year = int(df_epi["Anio"].iloc[-1])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card(
            "Pacientes", f"{total:,}", "Dataset calibrado", PALETTE["cyan"]
        )
    with c2:
        render_kpi_card(
            "Fallo Virologico",
            f"{fail_pct:.1f}%",
            f"{failures} casos (4:1)",
            PALETTE["magenta"],
        )
    with c3:
        render_kpi_card(
            f"Silueta (k={k_val})",
            f"{bundle['silhouette']:.4f}",
            "Separacion K-Means",
            PALETTE["yellow"],
        )
    with c4:
        render_kpi_card(
            "Exactitud RF",
            f"{bundle['accuracy'] * 100:.2f}%",
            f"F1 Fallo: {f1_fail:.1f}%",
            PALETTE["emerald"],
        )
    with c5:
        render_kpi_card(
            f"Cobertura TAR ({last_year})",
            f"{last_art:.1f}%",
            "Meta ONUSIDA: 95%",
            PALETTE["orange"],
        )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    tabs = st.tabs(
        [
            "Epidemiologia OMS",
            "Perfiles K-Means",
            "Modelo Random Forest",
            "API Stanford HIVdb",
            "Simulador Clinico",
        ]
    )
    with tabs[0]:
        render_epidemiology_tab(df_epi, epi_source)
    with tabs[1]:
        render_kmeans_tab(bundle, k_val)
    with tabs[2]:
        render_random_forest_tab(bundle, trees)
    with tabs[3]:
        render_stanford_tab(live_api)
    with tabs[4]:
        render_simulator_tab(bundle)

    render_footer()


if __name__ == "__main__":
    main()
