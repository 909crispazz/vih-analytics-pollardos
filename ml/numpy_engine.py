import numpy as np


class NumpyStandardScaler:
    def fit_transform(self, x_data):
        arr = np.asarray(x_data, dtype=float)
        self.mean_ = arr.mean(axis=0)
        self.scale_ = arr.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return (arr - self.mean_) / self.scale_

    def transform(self, x_data):
        arr = np.asarray(x_data, dtype=float)
        return (arr - self.mean_) / self.scale_


class NumpyKMeans:
    def __init__(self, n_clusters=3, random_state=42, n_init=10, max_iter=100):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

    def _fit_once(self, x_arr, rng):
        n_rows = x_arr.shape[0]
        centers = [x_arr[rng.randint(n_rows)]]
        for _ in range(1, self.n_clusters):
            dists = np.min(
                [np.sum((x_arr - c) ** 2, axis=1) for c in centers], axis=0
            )
            probs = dists / (dists.sum() + 1e-12)
            centers.append(x_arr[rng.choice(n_rows, p=probs)])
        centers = np.array(centers)

        for _ in range(self.max_iter):
            dists = np.linalg.norm(
                x_arr[:, None, :] - centers[None, :, :], axis=2
            )
            labels = np.argmin(dists, axis=1)
            updated = np.array(
                [
                    x_arr[labels == k].mean(axis=0)
                    if np.any(labels == k)
                    else centers[k]
                    for k in range(self.n_clusters)
                ]
            )
            if np.allclose(centers, updated, atol=1e-6):
                centers = updated
                break
            centers = updated
        inertia = float(np.sum((x_arr - centers[labels]) ** 2))
        return centers, labels, inertia

    def fit_predict(self, x_data):
        x_arr = np.asarray(x_data, dtype=float)
        rng = np.random.RandomState(self.random_state)
        best_inertia = np.inf
        best_centers, best_labels = None, None
        for _ in range(self.n_init):
            centers, labels, inertia = self._fit_once(x_arr, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        return best_labels

    def predict(self, x_data):
        x_arr = np.asarray(x_data, dtype=float)
        dists = np.linalg.norm(
            x_arr[:, None, :] - self.cluster_centers_[None, :, :], axis=2
        )
        return np.argmin(dists, axis=1)


def numpy_silhouette(x_data, labels):
    x_arr = np.asarray(x_data, dtype=float)
    sq_norm = np.sum(x_arr ** 2, axis=1, keepdims=True)
    dist_mat = np.sqrt(
        np.maximum(sq_norm + sq_norm.T - 2.0 * (x_arr @ x_arr.T), 0.0)
    )
    unique_lbls = np.unique(labels)
    scores = np.zeros(len(x_arr))
    for idx in range(len(x_arr)):
        own = labels == labels[idx]
        own_count = int(own.sum())
        a_val = (
            dist_mat[idx, own].sum() / (own_count - 1)
            if own_count > 1
            else 0.0
        )
        b_val = min(
            dist_mat[idx, labels == lbl].mean()
            for lbl in unique_lbls
            if lbl != labels[idx]
        )
        denom = max(a_val, b_val)
        scores[idx] = (b_val - a_val) / denom if denom > 0 else 0.0
    return float(scores.mean())


class _DecisionTree:
    def __init__(self, max_depth=8, min_leaf=2, max_features=2, rng=None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.rng = rng

    def fit(self, x_arr, y_arr, weights, n_features):
        self.importances = np.zeros(n_features, dtype=float)
        self.root = self._build(x_arr, y_arr, weights, 0)
        return self

    @staticmethod
    def _gini(w0, w1):
        total = w0 + w1
        if total <= 0:
            return 0.0
        p1 = w1 / total
        return 1.0 - p1 * p1 - (1.0 - p1) * (1.0 - p1)

    def _build(self, x_arr, y_arr, weights, depth):
        w1 = float(weights[y_arr == 1].sum())
        w0 = float(weights[y_arr == 0].sum())
        total = w0 + w1
        prob = w1 / total if total > 0 else 0.0
        if (
            depth >= self.max_depth
            or len(y_arr) < 2 * self.min_leaf
            or w0 == 0
            or w1 == 0
        ):
            return ("leaf", prob)

        parent_gini = self._gini(w0, w1)
        feats = self.rng.choice(
            x_arr.shape[1], size=self.max_features, replace=False
        )
        best_gain = 0.0
        best_split = None

        for feat in feats:
            col = x_arr[:, feat]
            cuts = np.unique(
                np.percentile(col, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            )
            for thr in cuts:
                left = col <= thr
                n_left = int(left.sum())
                n_right = len(y_arr) - n_left
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                wl = weights[left]
                yl = y_arr[left]
                wl1 = float(wl[yl == 1].sum())
                wl0 = float(wl.sum() - wl1)
                wr1 = w1 - wl1
                wr0 = w0 - wl0
                tl = wl0 + wl1
                tr = wr0 + wr1
                g_split = (tl / total) * self._gini(wl0, wl1) + (
                    tr / total
                ) * self._gini(wr0, wr1)
                gain = total * (parent_gini - g_split)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat, float(thr), left)

        if best_split is None:
            return ("leaf", prob)

        feat, thr, mask = best_split
        self.importances[feat] += best_gain
        left_node = self._build(
            x_arr[mask], y_arr[mask], weights[mask], depth + 1
        )
        right_node = self._build(
            x_arr[~mask], y_arr[~mask], weights[~mask], depth + 1
        )
        return ("node", feat, thr, left_node, right_node)

    @staticmethod
    def _predict_row(row, node):
        while node[0] == "node":
            _, feat, thr, left_n, right_n = node
            node = left_n if row[feat] <= thr else right_n
        return node[1]

    def predict_proba(self, x_arr):
        return np.array([self._predict_row(r, self.root) for r in x_arr])


class NumpyRandomForest:
    def __init__(
        self, n_estimators=100, random_state=42, class_weight="balanced"
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.class_weight = class_weight

    def fit(self, x_data, y_data):
        x_arr = np.asarray(x_data, dtype=float)
        y_arr = np.asarray(y_data, dtype=int)
        n_rows, n_cols = x_arr.shape
        rng = np.random.RandomState(self.random_state)
        max_f = max(1, int(np.sqrt(n_cols)))

        if self.class_weight == "balanced":
            n0 = max(1, int((y_arr == 0).sum()))
            n1 = max(1, int((y_arr == 1).sum()))
            weights = np.where(
                y_arr == 1, n_rows / (2.0 * n1), n_rows / (2.0 * n0)
            )
        else:
            weights = np.ones(n_rows, dtype=float)

        self.trees_ = []
        imp_sum = np.zeros(n_cols, dtype=float)
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n_rows, size=n_rows)
            tree = _DecisionTree(
                max_depth=8, min_leaf=2, max_features=max_f, rng=rng
            )
            tree.fit(x_arr[idx], y_arr[idx], weights[idx], n_cols)
            self.trees_.append(tree)
            imp_sum += tree.importances

        total_imp = imp_sum.sum()
        self.feature_importances_ = (
            imp_sum / total_imp if total_imp > 0 else np.ones(n_cols) / n_cols
        )
        return self

    def predict_proba(self, x_data):
        x_arr = np.asarray(x_data, dtype=float)
        p1 = np.mean([t.predict_proba(x_arr) for t in self.trees_], axis=0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, x_data):
        return (self.predict_proba(x_data)[:, 1] >= 0.5).astype(int)


def numpy_stratified_split(x_df, y_ser, test_size=0.25, random_state=42):
    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y_ser)
    train_idx, test_idx = [], []
    for cls in np.unique(y_arr):
        idx = np.where(y_arr == cls)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * test_size))
        test_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])
    return (
        x_df.iloc[train_idx],
        x_df.iloc[test_idx],
        y_ser.iloc[train_idx],
        y_ser.iloc[test_idx],
    )


def numpy_confusion_matrix(y_true, y_pred):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tp = int(((yt == 1) & (yp == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def numpy_classification_report(y_true, y_pred):
    cm = numpy_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    s0, s1 = int(tn + fp), int(fn + tp)
    tot = max(1, s0 + s1)

    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / s0 if s0 > 0 else 0.0
    f0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0

    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / s1 if s1 > 0 else 0.0
    f1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0

    return {
        "Controlado (0)": {
            "precision": p0,
            "recall": r0,
            "f1-score": f0,
            "support": s0,
        },
        "Fallo Virologico (1)": {
            "precision": p1,
            "recall": r1,
            "f1-score": f1,
            "support": s1,
        },
        "accuracy": (tn + tp) / tot,
        "macro avg": {
            "precision": (p0 + p1) / 2.0,
            "recall": (r0 + r1) / 2.0,
            "f1-score": (f0 + f1) / 2.0,
            "support": tot,
        },
        "weighted avg": {
            "precision": (p0 * s0 + p1 * s1) / tot,
            "recall": (r0 * s0 + r1 * s1) / tot,
            "f1-score": (f0 * s0 + f1 * s1) / tot,
            "support": tot,
        },
    }


def numpy_roc_auc(y_true, y_score):
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    order = np.argsort(-ys)
    y_ord = yt[order]
    pos = max(1, int((yt == 1).sum()))
    neg = max(1, int((yt == 0).sum()))
    tpr = np.concatenate([[0.0], np.cumsum(y_ord == 1) / pos])
    fpr = np.concatenate([[0.0], np.cumsum(y_ord == 0) / neg])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
