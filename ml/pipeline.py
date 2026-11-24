import numpy as np
import pandas as pd

from config.settings import CLUSTER_LABELS, KM_FEATURES, RF_FEATURES
from ml.numpy_engine import (
    NumpyKMeans,
    NumpyRandomForest,
    NumpyStandardScaler,
    numpy_classification_report,
    numpy_confusion_matrix,
    numpy_roc_auc,
    numpy_silhouette,
    numpy_stratified_split,
)

try:
    from sklearn.cluster import KMeans as SkKMeans
    from sklearn.ensemble import RandomForestClassifier as SkRandomForest
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        silhouette_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler as SkStandardScaler

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def train_models(df_base, n_clusters=3, n_estimators=100):
    df = df_base.copy()
    target = df["fallo_virologico"]

    if HAS_SKLEARN:
        engine = "Scikit-Learn"
        scaler = SkStandardScaler()
        x_scaled = scaler.fit_transform(df[KM_FEATURES])
        kmeans = SkKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        raw_clusters = kmeans.fit_predict(x_scaled)
        silhouette = float(silhouette_score(x_scaled, raw_clusters))

        x_tr, x_te, y_tr, y_te = train_test_split(
            df[RF_FEATURES],
            target,
            test_size=0.25,
            random_state=42,
            stratify=target,
        )
        rf = SkRandomForest(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(x_tr, y_tr)
        y_pred = rf.predict(x_te)
        y_prob = rf.predict_proba(x_te)[:, 1]
        acc = float(accuracy_score(y_te, y_pred))
        cm = confusion_matrix(y_te, y_pred)
        report = classification_report(
            y_te,
            y_pred,
            target_names=["Controlado (0)", "Fallo Virologico (1)"],
            output_dict=True,
        )
        roc_auc = float(roc_auc_score(y_te, y_prob))
    else:
        engine = "Motor NumPy Nativo"
        scaler = NumpyStandardScaler()
        x_scaled = scaler.fit_transform(df[KM_FEATURES])
        kmeans = NumpyKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        raw_clusters = kmeans.fit_predict(x_scaled)
        silhouette = numpy_silhouette(x_scaled, raw_clusters)

        x_tr, x_te, y_tr, y_te = numpy_stratified_split(
            df[RF_FEATURES], target, test_size=0.25, random_state=42
        )
        rf = NumpyRandomForest(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(x_tr, y_tr)
        y_pred = rf.predict(x_te)
        y_prob = rf.predict_proba(x_te)[:, 1]
        acc = float((y_pred == y_te.values).mean())
        cm = numpy_confusion_matrix(y_te.values, y_pred)
        report = numpy_classification_report(y_te.values, y_pred)
        roc_auc = numpy_roc_auc(y_te.values, y_prob)

    df["cluster_id"] = raw_clusters
    order = (
        df.groupby("cluster_id")["cd4_basal"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if n_clusters == 3:
        labels = CLUSTER_LABELS
    else:
        labels = [f"Perfil {i + 1}" for i in range(n_clusters)]
    cluster_map = {cid: labels[idx] for idx, cid in enumerate(order)}
    df["perfil_cluster"] = df["cluster_id"].map(cluster_map)
    probs_all = rf.predict_proba(df[RF_FEATURES])[:, 1] * 100.0
    df["prob_fallo_ia_pct"] = np.round(probs_all, 1)

    return {
        "engine": engine,
        "df": df,
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_map": cluster_map,
        "silhouette": silhouette,
        "rf": rf,
        "accuracy": acc,
        "cm": cm,
        "report": report,
        "roc_auc": roc_auc,
    }


def predict_patient(model_bundle, values):
    patient_df = pd.DataFrame([values], columns=RF_FEATURES)
    prob = float(model_bundle["rf"].predict_proba(patient_df)[0, 1] * 100.0)
    km_vec = model_bundle["scaler"].transform(patient_df[KM_FEATURES])
    cid = int(model_bundle["kmeans"].predict(km_vec)[0])
    profile = model_bundle["cluster_map"][cid]
    return prob, profile
