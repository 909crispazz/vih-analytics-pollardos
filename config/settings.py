PALETTE = {
    "indigo": "#240046",
    "purple": "#5A189A",
    "magenta": "#E81E63",
    "orange": "#FF6D00",
    "yellow": "#FFD100",
    "cyan": "#00B4D8",
    "emerald": "#06D6A0",
    "cream": "#FFFDF5",
    "white": "#FFFFFF",
}

CLUSTER_LABELS = [
    "Controlados (Supresion Viral)",
    "Riesgo Intermedio (Adherencia Irregular)",
    "Alto Riesgo (Fallo / Criterio SIDA)",
]

CLUSTER_COLORS = {
    "Controlados (Supresion Viral)": "#06D6A0",
    "Riesgo Intermedio (Adherencia Irregular)": "#FF6D00",
    "Alto Riesgo (Fallo / Criterio SIDA)": "#E81E63",
}

KM_FEATURES = [
    "cd4_basal",
    "carga_viral_log10",
    "adherencia_tar_pct",
    "score_resistencia_stanford",
]

RF_FEATURES = KM_FEATURES + ["edad", "meses_en_tar"]

FEATURE_LABELS = {
    "carga_viral_log10": "Carga Viral (log10)",
    "cd4_basal": "Linfocitos T CD4+ Basal",
    "adherencia_tar_pct": "Adherencia TAR (%)",
    "score_resistencia_stanford": "Resistencia Stanford",
    "meses_en_tar": "Meses en TAR",
    "edad": "Edad del Paciente",
}

TEAM_MEMBERS = [
    "Cristhian Ivan Sierra Contreras",
    "Juan Pablo Arredondo Escobar",
    "Lezzra Jahir Rocha Sainez",
    "Cristobal Enrique Paz Dominguez",
]

SIMULATOR_DEFAULTS = {
    "sim_cd4": 420,
    "sim_vl": 2.8,
    "sim_adh": 78.0,
    "sim_res": 20,
    "sim_edad": 34,
    "sim_meses": 24,
}

SIMULATOR_PRESETS = {
    "controlado": (680, 1.5, 96.0, 0, 35, 52),
    "moderado": (350, 3.2, 72.0, 25, 30, 18),
    "critico": (140, 5.1, 48.0, 75, 43, 12),
}

MUTATION_RULES = {
    "K103N": {"EFV": 60, "NVP": 60, "RPV": 0, "DOR": 0},
    "M184V": {"3TC": 60, "FTC": 60, "TDF": -10, "ABC": 15},
    "K65R": {"3TC": 15, "FTC": 15, "TDF": 60, "ABC": 45},
    "Y181C": {"EFV": 30, "NVP": 60, "RPV": 45, "DOR": 10},
    "G190A": {"EFV": 45, "NVP": 60, "RPV": 0, "DOR": 15},
    "L100I": {"EFV": 45, "NVP": 30, "RPV": 45, "DOR": 15},
    "M41L": {"TDF": 10, "ABC": 10},
    "T215Y": {"TDF": 15, "ABC": 15},
}

DRUG_CATALOG = {
    "3TC": ("Lamivudina (3TC)", "NRTI"),
    "FTC": ("Emtricitabina (FTC)", "NRTI"),
    "TDF": ("Tenofovir (TDF)", "NRTI"),
    "ABC": ("Abacavir (ABC)", "NRTI"),
    "EFV": ("Efavirenz (EFV)", "NNRTI"),
    "NVP": ("Nevirapina (NVP)", "NNRTI"),
    "RPV": ("Rilpivirina (RPV)", "NNRTI"),
    "DOR": ("Doravirina (DOR)", "NNRTI"),
}
