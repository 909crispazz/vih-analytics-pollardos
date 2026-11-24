from pathlib import Path
import streamlit as st

from config.settings import PALETTE, TEAM_MEMBERS


def load_stylesheet():
    root = Path(__file__).resolve().parent.parent
    css_path = root / "assets" / "styles.css"
    if css_path.exists():
        css_text = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-brand'>"
            "<div class='sidebar-title'>VIH-ANALYTICS</div>"
            "<div class='sidebar-sub'>"
            "Equipo Pollardos | TecNM ITL<br>"
            "Inteligencia Artificial | Grupo 5A"
            "</div></div><hr class='sidebar-divider'>",
            unsafe_allow_html=True,
        )
        st.markdown("### Controles de Modelos IA")
        k_val = st.slider("Clusters K-Means (k)", 2, 5, 3, 1)
        trees = st.slider("Arboles Random Forest", 50, 200, 100, 25)
        live_api = st.toggle("APIs Externas en Vivo", value=True)
        st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
        st.markdown("### Equipo Pollardos")
        members_html = "<br>".join(f"- {name}" for name in TEAM_MEMBERS)
        st.markdown(
            f"<div class='sidebar-team'>{members_html}<br><br>"
            "<b>Docente:</b><br>Mtra. Baltazar Flores Maria del Rosario</div>",
            unsafe_allow_html=True,
        )
    return k_val, trees, live_api


def render_hero(engine_name):
    st.markdown(
        "<div class='hero-pop'>"
        "<span class='hero-badge'>TEMA 6 | PROYECTO FINAL IA</span>"
        "<span class='hero-badge'>METAS 95-95-95 ONUSIDA</span>"
        "<span class='hero-badge'>INSTITUTO TECNOLOGICO DE LEON</span>"
        "<h1>VIH-Analytics Studio</h1>"
        "<p>Sistema Inteligente de Analisis Clinico y Epidemiologico de VIH "
        f"(K-Means + Random Forest | {engine_name}).</p></div>",
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value, subtitle, accent):
    html = (
        f"<div class='kpi-card' style='--accent:{accent};'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{subtitle}</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_callout(text_html, color_key="cyan"):
    bar = PALETTE.get(color_key, PALETTE["cyan"])
    st.markdown(
        f"<div class='callout-pop' style='--bar-color:{bar};'>"
        f"{text_html}</div>",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        "<div class='footer-pop'>"
        "<b>VIH-Analytics Studio</b> | Equipo Pollardos | ITL | "
        "<i>Prototipo sobre dataset sintetico calibrado (n=1,000); "
        "siguiente fase: validacion externa ACTG 175 (n=2,139).</i></div>",
        unsafe_allow_html=True,
    )
