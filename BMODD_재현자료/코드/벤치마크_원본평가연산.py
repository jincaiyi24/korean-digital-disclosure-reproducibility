"""기존 인간참조 평가에서 추출한 동일 연산. 입력 경로와 공개 명칭만 외부 진입점에서 처리한다."""
import numpy as np
LABELS=("BMODD","NonBMODD","Exclude")
STRATA=("BMODD","NonBMODD","candidate_Exclude","noncandidate_valid")
METRICS=("Accuracy","Macro_F1","BMODD_Precision","BMODD_Recall","BMODD_F1","NonBMODD_Precision","NonBMODD_Recall","NonBMODD_F1","Exclude_Precision","Exclude_Recall","Exclude_F1")

def metrics_from_matrices(matrices):
    """Last two dimensions: human rows and prediction columns. Return 11 metrics."""
    cm = np.asarray(matrices, dtype=np.float64)
    if cm.shape[-2:] != (3, 3) or not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("Require finite nonnegative 3x3 confusion matrices")
    true_support = cm.sum(axis=-1); pred_support = cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support != 0)
    recall = np.divide(tp, true_support, out=np.zeros_like(tp), where=true_support != 0)
    denom = true_support + pred_support
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0)
    total = cm.sum(axis=(-2, -1))
    accuracy = np.divide(tp.sum(axis=-1), total, out=np.zeros_like(total), where=total != 0)
    pieces = [accuracy, f1.mean(axis=-1)]
    for index in range(3): pieces.extend((precision[..., index], recall[..., index], f1[..., index]))
    return np.stack(pieces, axis=-1)

def confusion_matrices(gold, predictions, weights):
    gold = np.asarray(gold, dtype=np.int64); pred = np.asarray(predictions, dtype=np.int64)
    weight = np.asarray(weights, dtype=np.float64)
    if gold.ndim != 1 or pred.ndim != 2 or pred.shape[1] != len(gold) or weight.shape != gold.shape:
        raise ValueError("Aligned one-dimensional gold/weights and methods-by-sentences predictions required")
    if not np.isin(gold, (0, 1, 2)).all() or not np.isin(pred, (0, 1, 2)).all():
        raise ValueError("Class indices outside 0..2")
    if not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Finite nonnegative design weights required")
    keys = pred + 3 * gold[np.newaxis, :] + 9 * np.arange(len(pred))[:, np.newaxis]
    return np.bincount(keys.ravel(), weights=np.broadcast_to(weight, pred.shape).ravel(),
                       minlength=len(pred) * 9).reshape(len(pred), 3, 3)

def aggregate_seed_metrics(values):
    """Average already-computed seed metrics; never average predictions or matrices."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (9, len(METRICS)):
        raise ValueError("Exactly rules plus four BERT plus four FinBERT metrics required")
    return np.stack((values[0], values[1:5].mean(axis=0), values[5:9].mean(axis=0)))

def make_resample_indices(stratum_values, replicates=5000, seed=20261025):
    values = np.asarray(stratum_values)
    if len(values) != 300 or set(values.tolist()) != set(STRATA):
        raise ValueError("The four original strata must cover all fixed300")
    positions = [np.flatnonzero(values == stratum) for stratum in STRATA]
    if any(len(position) != 75 for position in positions):
        raise ValueError("Each original stratum must contain exactly75")
    rng = np.random.default_rng(seed)
    return np.stack([np.concatenate([rng.choice(position, 75, replace=True) for position in positions])
                     for _ in range(replicates)]).astype(np.int16)

def stratified_bootstrap(gold, predictions, weights, stratum_values, replicates=5000, seed=20261025):
    indices = make_resample_indices(stratum_values, replicates, seed)
    seed_metrics = np.empty((2, replicates, 9, len(METRICS)), dtype=np.float64)
    group_metrics = np.empty((2, replicates, 3, len(METRICS)), dtype=np.float64)
    differences = np.empty((2, replicates, 2), dtype=np.float64)
    for repeat, index in enumerate(indices):
        for scheme, row_weights in enumerate((np.ones(300), weights)):
            matrix = confusion_matrices(gold[index], predictions[:, index], row_weights[index])
            seed_metrics[scheme, repeat] = metrics_from_matrices(matrix)
            grouped = aggregate_seed_metrics(seed_metrics[scheme, repeat])
            group_metrics[scheme, repeat] = grouped
            differences[scheme, repeat] = grouped[1:, 1] - grouped[0, 1]
    if not all(np.isfinite(array).all() for array in (seed_metrics, group_metrics, differences)):
        raise ValueError("Bootstrap returned a nonfinite metric")
    return indices, seed_metrics, group_metrics, differences

def calculate_results(data):
    point_matrices = []; point_seed = []; point_groups = []
    for row_weights in (np.ones(300), data["weights"]):
        matrix = confusion_matrices(data["gold"], data["predictions"], row_weights)
        values = metrics_from_matrices(matrix)
        point_matrices.append(matrix); point_seed.append(values); point_groups.append(aggregate_seed_metrics(values))
    indices, boot_seed, boot_groups, boot_diff = stratified_bootstrap(
        data["gold"], data["predictions"], data["weights"], data["strata"], 5000, 20261025)
    return {"point_matrices": np.stack(point_matrices), "point_seed": np.stack(point_seed),
            "point_groups": np.stack(point_groups), "bootstrap_indices": indices,
            "bootstrap_seed_metrics": boot_seed, "bootstrap_group_metrics": boot_groups,
            "bootstrap_differences": boot_diff}

