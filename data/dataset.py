import numpy as np
import pandas as pd


def _build_group(rng, size, specs):
    return pd.DataFrame(
        {
            "cd4_basal": np.clip(
                rng.normal(specs[0], specs[1], size), 20, 1200
            ).round(0),
            "carga_viral_log10": np.clip(
                rng.normal(specs[2], specs[3], size), 1.3, 6.5
            ).round(2),
            "adherencia_tar_pct": np.clip(
                rng.normal(specs[4], specs[5], size), 20, 100
            ).round(1),
            "score_resistencia_stanford": np.clip(
                rng.normal(specs[6], specs[7], size), 0, 120
            ).round(0),
            "edad": np.clip(
                rng.normal(specs[8], specs[9], size), 18, 75
            ).round(0),
            "meses_en_tar": np.clip(
                rng.normal(specs[10], specs[11], size), 3, 180
            ).round(0),
        }
    )


def generate_clinical_dataset(seed=42):
    rng = np.random.RandomState(seed)
    group_a = _build_group(
        rng, 620, (640, 110, 1.65, 0.25, 95.0, 4.0, 4.0, 5.0, 36, 9, 48, 22)
    )
    group_b = _build_group(
        rng, 210, (380, 75, 3.05, 0.45, 76.0, 7.0, 22.0, 10.0, 32, 8, 24, 14)
    )
    group_c = _build_group(
        rng, 170, (155, 55, 4.85, 0.55, 52.0, 11.0, 68.0, 16.0, 41, 11, 14, 9)
    )

    df = pd.concat([group_a, group_b, group_c], ignore_index=True)
    logit = (
        -0.006 * df["cd4_basal"]
        + 1.15 * df["carga_viral_log10"]
        - 0.085 * df["adherencia_tar_pct"]
        + 0.045 * df["score_resistencia_stanford"]
        + 2.3
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    df["fallo_virologico"] = (rng.rand(len(df)) < prob).astype(int)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    ids = [f"VIH-{idx:04d}" for idx in range(1, len(df) + 1)]
    df.insert(0, "ID_PACIENTE", ids)
    return df
