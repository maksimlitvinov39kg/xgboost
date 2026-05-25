"""Benchmark: CPU multilabel macro-AUC on synthetic data."""
import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import roc_auc_score


def bench_xgboost_builtin(dtrain, deval, num_boost_round=5):
    """Time the built-in multilabel AUC inside XGBoost training."""
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "eval_metric": "auc",
        "seed": 42,
        "max_depth": 4,
        "eta": 0.3,
    }
    evals = [(deval, "eval")]

    # Warm-up (JIT / cache)
    xgb.train(params, dtrain, num_boost_round=1, evals=evals, verbose_eval=False)

    t0 = time.perf_counter()
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, verbose_eval=False)
    t1 = time.perf_counter()
    return t1 - t0, bst


def bench_sklearn_macro_auc(y_true, y_pred):
    """Time sklearn macro-average ROC-AUC (CPU, numpy)."""
    # Warm-up
    roc_auc_score(y_true, y_pred, average="macro")

    t0 = time.perf_counter()
    auc = roc_auc_score(y_true, y_pred, average="macro")
    t1 = time.perf_counter()
    return t1 - t0, auc


def run_one(n_samples, n_features, n_targets, n_labels):
    print(f"\n=== Config: n_samples={n_samples:,}, n_features={n_features}, "
          f"n_targets={n_targets}, n_labels={n_labels} ===")

    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_targets,
        n_labels=n_labels,
        random_state=42,
    )

    # Train / eval split
    split = int(n_samples * 0.8)
    X_train, X_eval = X[:split], X[split:]
    y_train, y_eval = y[:split], y[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    # 1) Built-in XGBoost multilabel AUC (includes tree building + eval)
    t_xgb_total, bst = bench_xgboost_builtin(dtrain, deval, num_boost_round=5)
    print(f"XGBoost total train+eval (5 rounds): {t_xgb_total:.3f}s")

    # 2) Isolated metric time: predict then sklearn macro-AUC
    preds = bst.predict(deval)
    t_sklearn, sk_auc = bench_sklearn_macro_auc(y_eval, preds)
    print(f"sklearn macro-AUC (predict excluded): {t_sklearn:.3f}s  |  AUC={sk_auc:.4f}")

    # 3) Predict-only time (this is the GPU->CPU copy bottleneck on CUDA)
    t0 = time.perf_counter()
    _ = bst.predict(deval)
    t_predict = time.perf_counter() - t0
    print(f"XGBoost predict() alone:            {t_predict:.3f}s")

    # 4) Estimate eval metric-only time using a dummy booster (0 trees)
    params_eval = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "eval_metric": "auc",
        "seed": 42,
    }
    dummy = xgb.train(params_eval, deval, num_boost_round=0, verbose_eval=False)
    t0 = time.perf_counter()
    _ = dummy.eval_set([(deval, "eval")], iteration=0)
    t_metric_only = time.perf_counter() - t0
    print(f"XGBoost metric-only (0 trees):      {t_metric_only:.3f}s")

    return {
        "n_samples": n_samples,
        "n_targets": n_targets,
        "t_xgb_total": t_xgb_total,
        "t_sklearn": t_sklearn,
        "t_predict": t_predict,
        "t_metric_only": t_metric_only,
    }


def main():
    results = []

    # Small: what fits easily in memory
    results.append(run_one(n_samples=100_000, n_features=32, n_targets=4, n_labels=2))

    # Medium
    results.append(run_one(n_samples=500_000, n_features=32, n_targets=4, n_labels=2))

    # Large (1M) — depending on RAM
    results.append(run_one(n_samples=1_000_000, n_features=32, n_targets=4, n_labels=2))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for r in results:
        print(f"n={r['n_samples']:,}  "
              f"xgb_total={r['t_xgb_total']:.3f}s  "
              f"sklearn_auc={r['t_sklearn']:.3f}s  "
              f"predict={r['t_predict']:.3f}s  "
              f"metric_only={r['t_metric_only']:.3f}s")

    print("\nNote: On a CUDA GPU the 'metric_only' time should drop to ~0.01-0.05s")
    print("      because the metric stays on device (no predict() / memcpy).")


if __name__ == "__main__":
    main()
