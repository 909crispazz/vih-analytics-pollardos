import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import (
    CLUSTER_COLORS,
    FEATURE_LABELS,
    PALETTE,
    RF_FEATURES,
)


def apply_chart_theme(fig, title=None, height=400, top_margin=58):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family="Fredoka, sans-serif",
                size=17,
                color=PALETTE["indigo"],
            ),
            x=0.02,
            y=0.95,
        )
        if title
        else None,
        paper_bgcolor=PALETTE["white"],
        plot_bgcolor="#FFFDF8",
        font=dict(
            family="DM Sans, sans-serif",
            color=PALETTE["indigo"],
            size=12,
        ),
        height=height,
        margin=dict(l=45, r=30, t=top_margin if title else 35, b=45),
        legend=dict(
            bgcolor="rgba(255,253,245,0.95)",
            bordercolor=PALETTE["indigo"],
            borderwidth=1.2,
            font=dict(color=PALETTE["indigo"]),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(36,0,70,0.08)",
        linecolor=PALETTE["indigo"],
        tickfont=dict(color=PALETTE["indigo"]),
        title_font=dict(color=PALETTE["indigo"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(36,0,70,0.08)",
        linecolor=PALETTE["indigo"],
        tickfont=dict(color=PALETTE["indigo"]),
        title_font=dict(color=PALETTE["indigo"]),
    )
    return fig


def build_epidemiology_chart(df_epi):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df_epi["Anio"],
            y=df_epi["Personas_VIH"],
            name="Personas viviendo con VIH",
            marker=dict(
                color=PALETTE["cyan"],
                line=dict(color=PALETTE["indigo"], width=1.8),
            ),
            opacity=0.9,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_epi["Anio"],
            y=df_epi["Cobertura_TAR_Pct"],
            name="Cobertura TAR (%)",
            mode="lines+markers+text",
            text=[f"{v:.0f}%" for v in df_epi["Cobertura_TAR_Pct"]],
            textposition="top center",
            textfont=dict(color=PALETTE["indigo"], size=11),
            line=dict(color=PALETTE["magenta"], width=4),
            marker=dict(
                size=10,
                color=PALETTE["yellow"],
                line=dict(color=PALETTE["indigo"], width=2),
            ),
        ),
        secondary_y=True,
    )
    fig.add_hline(
        y=95,
        line_dash="dot",
        line_color=PALETTE["emerald"],
        line_width=2.5,
        annotation_text="Meta ONUSIDA 95%",
        annotation_font_color=PALETTE["indigo"],
        secondary_y=True,
    )
    fig = apply_chart_theme(
        fig,
        "Mexico: Prevalencia de VIH vs. Cobertura TAR (%)",
        height=430,
    )
    fig.update_yaxes(title_text="Personas con VIH", secondary_y=False)
    fig.update_yaxes(
        title_text="Cobertura TAR (%)", range=[0, 108], secondary_y=True
    )
    return fig


def build_kmeans_3d_chart(df_mod, k_val):
    seq = [
        PALETTE["emerald"],
        PALETTE["orange"],
        PALETTE["magenta"],
        PALETTE["cyan"],
        PALETTE["purple"],
    ]
    cmap = CLUSTER_COLORS if k_val == 3 else None
    fig = px.scatter_3d(
        df_mod,
        x="cd4_basal",
        y="carga_viral_log10",
        z="adherencia_tar_pct",
        color="perfil_cluster",
        color_discrete_map=cmap,
        color_discrete_sequence=seq,
        hover_data=["ID_PACIENTE", "score_resistencia_stanford"],
        opacity=0.82,
        labels={
            "cd4_basal": "CD4 Basal",
            "carga_viral_log10": "Carga Viral (log10)",
            "adherencia_tar_pct": "Adherencia (%)",
            "perfil_cluster": "Perfil K-Means",
        },
    )
    fig.update_traces(
        marker=dict(size=4.5, line=dict(width=0.4, color=PALETTE["indigo"]))
    )
    return apply_chart_theme(fig, "Segmentacion Clinica 3D (K-Means)", 440)


def build_cd4_violin_chart(df_mod, k_val):
    seq = [
        PALETTE["emerald"],
        PALETTE["orange"],
        PALETTE["magenta"],
        PALETTE["cyan"],
        PALETTE["purple"],
    ]
    cmap = CLUSTER_COLORS if k_val == 3 else None
    fig = px.violin(
        df_mod,
        x="perfil_cluster",
        y="cd4_basal",
        color="perfil_cluster",
        color_discrete_map=cmap,
        color_discrete_sequence=seq,
        box=True,
        labels={
            "perfil_cluster": "Perfil",
            "cd4_basal": "CD4 Basal (cel/mm3)",
        },
    )
    fig.add_hline(
        y=200,
        line_dash="dash",
        line_color=PALETTE["magenta"],
        line_width=2.5,
        annotation_text="Criterio SIDA (CD4 < 200)",
        annotation_font_color=PALETTE["indigo"],
    )
    fig = apply_chart_theme(fig, "Distribucion de CD4+ por Perfil", 440)
    fig.update_layout(showlegend=False)
    return fig


def build_confusion_matrix_chart(cm):
    labels = ["Controlado (0)", "Fallo Virologico (1)"]
    text = [
        [f"VN: {cm[0, 0]}", f"FP: {cm[0, 1]}"],
        [f"FN: {cm[1, 0]}", f"VP: {cm[1, 1]}"],
    ]
    scale = [
        [0, "#FFFDF5"],
        [0.45, PALETTE["cyan"]],
        [1, PALETTE["purple"]],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=16, family="Fredoka"),
            colorscale=scale,
            showscale=False,
        )
    )
    fig = apply_chart_theme(fig, "Matriz de Confusion (Prueba n=250)", 390)
    fig.update_xaxes(title_text="Prediccion")
    fig.update_yaxes(title_text="Clase Real", autorange="reversed")
    return fig


def build_feature_importance_chart(rf_model):
    df_imp = pd.DataFrame(
        {
            "Variable": [FEATURE_LABELS[col] for col in RF_FEATURES],
            "Importancia": rf_model.feature_importances_ * 100.0,
        }
    ).sort_values("Importancia")
    colors = [
        PALETTE["purple"],
        PALETTE["cyan"],
        PALETTE["emerald"],
        PALETTE["yellow"],
        PALETTE["orange"],
        PALETTE["magenta"],
    ]
    fig = go.Figure(
        go.Bar(
            x=df_imp["Importancia"],
            y=df_imp["Variable"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color=PALETTE["indigo"], width=1.8),
            ),
            text=[f"{v:.1f}%" for v in df_imp["Importancia"]],
            textposition="outside",
            textfont=dict(color=PALETTE["indigo"]),
        )
    )
    fig = apply_chart_theme(fig, "Importancia de Variables (%)", 390)
    max_val = float(df_imp["Importancia"].max())
    fig.update_xaxes(title_text="Peso (%)", range=[0, max_val + 8])
    return fig


def build_stanford_chart(df_stf, mutations):
    def pick_color(score):
        if score >= 60:
            return PALETTE["magenta"]
        if score >= 30:
            return PALETTE["orange"]
        if score >= 15:
            return PALETTE["yellow"]
        return PALETTE["emerald"]

    fig = go.Figure(
        go.Bar(
            x=df_stf["Score"],
            y=df_stf["Farmaco"],
            orientation="h",
            marker=dict(
                color=[pick_color(s) for s in df_stf["Score"]],
                line=dict(color=PALETTE["indigo"], width=1.8),
            ),
            text=[
                f"{int(s)} pts ({lvl})"
                for s, lvl in zip(df_stf["Score"], df_stf["Nivel"])
            ],
            textposition="outside",
            textfont=dict(color=PALETTE["indigo"]),
        )
    )
    fig.add_vline(
        x=60,
        line_dash="dash",
        line_color=PALETTE["magenta"],
        annotation_text="Resistencia Alta (60)",
        annotation_font_color=PALETTE["indigo"],
    )
    title = "Resistencia por Farmaco (" + ", ".join(mutations) + ")"
    fig = apply_chart_theme(fig, title, 410)
    x_min = min(-15, float(df_stf["Score"].min()) - 10)
    x_max = max(95, float(df_stf["Score"].max()) + 38)
    fig.update_xaxes(title_text="Score Stanford HIVdb", range=[x_min, x_max])
    return fig


def build_risk_gauge(prob_pct, alert_color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={
                "suffix": "%",
                "font": {
                    "family": "Fredoka",
                    "size": 46,
                    "color": PALETTE["indigo"],
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": PALETTE["indigo"],
                    "tickfont": {"color": PALETTE["indigo"]},
                },
                "bar": {
                    "color": alert_color,
                    "line": {"color": PALETTE["indigo"], "width": 2},
                },
                "bgcolor": PALETTE["cream"],
                "borderwidth": 2.5,
                "bordercolor": PALETTE["indigo"],
                "steps": [
                    {"range": [0, 30], "color": "#B8F7E3"},
                    {"range": [30, 60], "color": "#FFE999"},
                    {"range": [60, 100], "color": "#FFD6E0"},
                ],
            },
        )
    )
    return apply_chart_theme(
        fig,
        "Probabilidad de Fallo Virologico (Random Forest)",
        height=310,
        top_margin=72,
    )
