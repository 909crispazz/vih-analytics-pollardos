import pandas as pd
import requests

from config.settings import DRUG_CATALOG, MUTATION_RULES

WB_BASE = "https://api.worldbank.org/v2/country"
STANFORD_URL = "https://hivdb.stanford.edu/graphql"

FALLBACK_HIV = [
    (2014, 210000),
    (2015, 230000),
    (2016, 245000),
    (2017, 260000),
    (2018, 280000),
    (2019, 300000),
    (2020, 320000),
    (2021, 340000),
    (2022, 360000),
    (2023, 370000),
]

FALLBACK_ART = [
    (2014, 52.0),
    (2015, 57.0),
    (2016, 61.0),
    (2017, 65.0),
    (2018, 69.0),
    (2019, 73.0),
    (2020, 75.0),
    (2021, 78.0),
    (2022, 81.0),
    (2023, 84.0),
]


def _fetch_wb_series(country, indicator, col_name):
    url = f"{WB_BASE}/{country}/indicator/{indicator}?format=json&per_page=100"
    resp = requests.get(url, timeout=4)
    rows = []
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list) and len(data) > 1 and data[1]:
            for item in data[1]:
                if item.get("date") and item.get("value") is not None:
                    rows.append(
                        {
                            "Anio": int(item["date"]),
                            col_name: float(item["value"]),
                        }
                    )
    return rows


def fetch_epidemiology(country="MEX", use_live_api=True):
    source = "Respaldo Oficial OMS / ONUSIDA / CENSIDA"
    hiv_rows, art_rows = [], []

    if use_live_api:
        try:
            hiv_rows = _fetch_wb_series(country, "SH.HIV.TOTL", "Personas_VIH")
            art_rows = _fetch_wb_series(
                country, "SH.HIV.ARTC.ZS", "Cobertura_TAR_Pct"
            )
            if hiv_rows and art_rows:
                source = "World Bank / UNAIDS REST API v2 (En Vivo)"
        except Exception:
            pass

    if not hiv_rows:
        hiv_rows = [
            {"Anio": year, "Personas_VIH": float(val)}
            for year, val in FALLBACK_HIV
        ]
    if not art_rows:
        art_rows = [
            {"Anio": year, "Cobertura_TAR_Pct": float(val)}
            for year, val in FALLBACK_ART
        ]

    df_hiv = pd.DataFrame(hiv_rows).sort_values("Anio")
    df_art = pd.DataFrame(art_rows).sort_values("Anio")
    merged = pd.merge(df_hiv, df_art, on="Anio", how="inner")
    return merged.tail(10).reset_index(drop=True), source


def _resistance_level(score):
    if score >= 60:
        return "High-Level Resistance"
    if score >= 30:
        return "Intermediate Resistance"
    if score >= 15:
        return "Low-Level Resistance"
    if score >= 10:
        return "Potential Low-Level Resistance"
    return "Susceptible"


def fetch_stanford_resistance(mutations, use_live_api=True):
    muts = list(mutations) if mutations else ["K103N"]
    mut_list = ", ".join(f'"RT:{m}"' for m in muts)
    query = (
        "query { viewer { mutationsAnalysis(mutations: ["
        + mut_list
        + "]) { drugResistance { drugScores { drug { name displayAbbr } "
        + "drugClass { name } score text } } } } }"
    )

    if use_live_api:
        try:
            resp = requests.post(
                STANFORD_URL,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code == 200:
                payload = resp.json()
                blocks = (
                    payload.get("data", {})
                    .get("viewer", {})
                    .get("mutationsAnalysis", {})
                    .get("drugResistance", [])
                )
                rows = []
                for block in blocks:
                    for item in block.get("drugScores", []):
                        abbr = item["drug"]["displayAbbr"]
                        drug_name = item["drug"]["name"]
                        if abbr in DRUG_CATALOG:
                            rows.append(
                                {
                                    "Codigo": abbr,
                                    "Farmaco": f"{drug_name} ({abbr})",
                                    "Clase": item["drugClass"]["name"],
                                    "Score": float(item["score"]),
                                    "Nivel": item["text"],
                                }
                            )
                if rows:
                    df_live = pd.DataFrame(rows).sort_values("Score")
                    return (
                        df_live,
                        "Stanford HIVdb GraphQL API (En Vivo)",
                        payload,
                    )
        except Exception:
            pass

    rows = []
    for abbr, (name, drug_class) in DRUG_CATALOG.items():
        total = sum(MUTATION_RULES.get(m, {}).get(abbr, 0) for m in muts)
        rows.append(
            {
                "Codigo": abbr,
                "Farmaco": name,
                "Clase": drug_class,
                "Score": float(total),
                "Nivel": _resistance_level(total),
            }
        )
    df_off = pd.DataFrame(rows).sort_values("Score")
    payload = {
        "source": "Stanford HIVdb v9 (Respaldo Calibrado)",
        "gene": "RT",
        "mutations": [f"RT:{m}" for m in muts],
        "drugScores": df_off[
            ["Farmaco", "Clase", "Score", "Nivel"]
        ].to_dict(orient="records"),
    }
    return df_off, "Respaldo Calibrado Stanford HIVdb v9", payload
