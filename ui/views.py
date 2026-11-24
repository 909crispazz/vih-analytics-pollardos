import json
import pandas as pd
import streamlit as st

from config.settings import (
    MUTATION_RULES,
    PALETTE,
    SIMULATOR_DEFAULTS,
    SIMULATOR_PRESETS,
)
from data.api_client import fetch_stanford_resistance
from ml.pipeline import predict_patient
from ui.charts import (
    build_cd4_violin_chart,
    build_confusion_matrix_chart,
    build_epidemiology_chart,
    build_feature_importance_chart,
    build_kmeans_3d_chart,
    build_risk_gauge,
    build_stanford_chart,
)
from ui.components import render_callout, render_kpi_card


def render_epidemiology_tab(df_epi, source):
    last_art = df_epi["Cobertura_TAR_Pct"].iloc[-1]
    render_callout(
        f"<b>Fuente (Capa A):</b> {source}. Al alcanzar <b>{last_art:.1f}%</b> "
        "de cobertura TAR, el aumento de personas viviendo con VIH refleja "
        "mayor supervivencia clinica y control viral.",
        "cyan",
    )
    col_left, col_right = st.columns([1.8, 1])
    with col_left:
        fig = build_epidemiology_chart(df_epi)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.markdown("#### Serie Historica Mexico")
        table = df_epi.rename(
            columns={
                "Anio": "Anio",
                "Personas_VIH": "Personas VIH",
                "Cobertura_TAR_Pct": "Cobertura TAR (%)",
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_kmeans_tab(bundle, k_val):
    df_mod = bundle["df"]
    render_callout(
        f"<b>Aprendizaje No Supervisado (k={k_val}):</b> Coeficiente de "
        f"Silueta <b>{bundle['silhouette']:.4f}</b> con estandarizacion "
        "Z-score. El perfil de Alto Riesgo cae debajo de 200 CD4 (SIDA).",
        "yellow",
    )
    col_a, col_b = st.columns([1.25, 1])
    with col_a:
        st.plotly_chart(
            build_kmeans_3d_chart(df_mod, k_val), use_container_width=True
        )
    with col_b:
        st.plotly_chart(
            build_cd4_violin_chart(df_mod, k_val), use_container_width=True
        )

    summary = (
        df_mod.groupby("perfil_cluster")
        .agg(
            Pacientes=("ID_PACIENTE", "count"),
            CD4=("cd4_basal", "mean"),
            Carga_Viral=("carga_viral_log10", "mean"),
            Adherencia=("adherencia_tar_pct", "mean"),
            Resistencia=("score_resistencia_stanford", "mean"),
            Fallo_Pct=("fallo_virologico", lambda s: s.mean() * 100.0),
        )
        .reset_index()
        .round(2)
    )
    summary.columns = [
        "Perfil Clinico",
        "Pacientes",
        "CD4 Prom.",
        "Carga Viral",
        "Adherencia (%)",
        "Resistencia",
        "Fallo (%)",
    ]
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_random_forest_tab(bundle, n_trees):
    render_callout(
        f"<b>Random Forest ({n_trees} arboles, class_weight='balanced'):</b> "
        f"Exactitud <b>{bundle['accuracy'] * 100:.2f}%</b> | "
        f"AUC ROC <b>{bundle['roc_auc']:.4f}</b> sobre 250 pacientes de "
        "prueba (division 75/25 estratificada).",
        "emerald",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            build_confusion_matrix_chart(bundle["cm"]),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            build_feature_importance_chart(bundle["rf"]),
            use_container_width=True,
        )

    rep_df = (
        pd.DataFrame(bundle["report"])
        .T.reset_index()
        .rename(columns={"index": "Clase"})
    )
    rep_df = rep_df[rep_df["Clase"] != "accuracy"].copy()
    cols = ["precision", "recall", "f1-score"]
    rep_df[cols] = rep_df[cols].round(4)
    rep_df["support"] = rep_df["support"].astype(int)
    st.dataframe(rep_df, use_container_width=True, hide_index=True)


def render_stanford_tab(use_live_api):
    render_callout(
        "<b>Capa B (Stanford HIVdb GraphQL):</b> Evalua mutaciones del gen "
        "<code>RT</code> para cuantificar resistencia por antirretroviral.",
        "magenta",
    )
    selected = st.multiselect(
        "Mutaciones en gen RT:",
        options=list(MUTATION_RULES.keys()),
        default=["K103N", "M184V"],
    )
    df_stf, src, raw_json = fetch_stanford_resistance(
        tuple(selected), use_live_api=use_live_api
    )
    max_score = float(df_stf["Score"].max())
    viable = df_stf[df_stf["Score"] < 15]["Codigo"].tolist()

    col_a, col_b = st.columns([1.6, 1])
    with col_a:
        st.plotly_chart(
            build_stanford_chart(df_stf, selected or ["K103N"]),
            use_container_width=True,
        )
    with col_b:
        st.caption(f"Servicio: {src}")
        accent = PALETTE["magenta"] if max_score >= 60 else PALETTE["emerald"]
        render_kpi_card(
            "Score Maximo",
            f"{max_score:.0f} pts",
            "Variable de entrada a Random Forest",
            accent,
        )
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        render_kpi_card(
            "Farmacos Susceptibles",
            ", ".join(viable) if viable else "Ninguno",
            "Opciones activas en panel RT",
            PALETTE["emerald"],
        )

    with st.expander("Ver respuesta JSON de Stanford HIVdb"):
        st.code(
            json.dumps(raw_json, indent=2, ensure_ascii=False)[:2200],
            language="json",
        )


def _apply_preset(preset_key):
    vals = SIMULATOR_PRESETS[preset_key]
    keys = list(SIMULATOR_DEFAULTS.keys())
    for key, val in zip(keys, vals):
        st.session_state[key] = val


def render_simulator_tab(bundle):
    render_callout(
        "<b>Simulador Clinico en Vivo:</b> Selecciona un escenario o mueve "
        "los controles para predecir riesgo de fallo virologico y perfil.",
        "orange",
    )
    for key, val in SIMULATOR_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val

    b1, b2, b3 = st.columns(3)
    with b1:
        st.button(
            "Caso 1: Paciente Controlado",
            use_container_width=True,
            on_click=_apply_preset,
            args=("controlado",),
        )
    with b2:
        st.button(
            "Caso 2: Adherencia Irregular",
            use_container_width=True,
            on_click=_apply_preset,
            args=("moderado",),
        )
    with b3:
        st.button(
            "Caso 3: Alerta Critica",
            use_container_width=True,
            on_click=_apply_preset,
            args=("critico",),
        )

    col_in, col_out = st.columns([1.05, 1.35])
    with col_in:
        cd4 = st.slider(
            "Linfocitos T CD4+ (cel/mm3)", 20, 1200, step=10, key="sim_cd4"
        )
        vl = st.slider("Carga Viral (log10)", 1.3, 6.5, step=0.1, key="sim_vl")
        adh = st.slider(
            "Adherencia TAR (%)", 20.0, 100.0, step=1.0, key="sim_adh"
        )
        res_s = st.slider(
            "Resistencia Stanford (pts)", 0, 120, step=5, key="sim_res"
        )
        edad = st.slider("Edad (anios)", 18, 75, step=1, key="sim_edad")
        meses = st.slider("Meses en TAR", 3, 180, step=1, key="sim_meses")

    prob, profile = predict_patient(bundle, [cd4, vl, adh, res_s, edad, meses])
    if prob >= 60:
        color_key, color_hex = "magenta", PALETTE["magenta"]
        verdict = (
            "ALTO RIESGO: Solicitar genotipificacion Stanford HIVdb y "
            "valorar cambio de esquema antirretroviral."
        )
    elif prob >= 30:
        color_key, color_hex = "orange", PALETTE["orange"]
        verdict = (
            "RIESGO MODERADO: Reforzar adherencia terapeutica y repetir "
            "carga viral en 8 a 12 semanas."
        )
    else:
        color_key, color_hex = "emerald", PALETTE["emerald"]
        verdict = (
            "CONTROL VIROLOGICO: Mantener esquema TAR actual y monitoreo "
            "semestral de rutina."
        )

    with col_out:
        st.plotly_chart(
            build_risk_gauge(prob, color_hex), use_container_width=True
        )
        render_callout(
            f"<b>Perfil K-Means:</b> {profile}<br><b>Dictamen:</b> {verdict}",
            color_key,
        )
