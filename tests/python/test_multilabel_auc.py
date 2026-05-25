"""Smoke test for multilabel macro-AUC metric."""
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import roc_auc_score


def test_multilabel_auc_cpu():
    """Test that built-in auc works for multi-output binary classification."""
    n_samples = 256
    n_features = 16
    n_targets = 4

    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_targets,
        n_labels=2,
        random_state=42,
    )

    # Split into train / eval
    split = int(n_samples * 0.8)
    X_train, X_eval = X[:split], X[split:]
    y_train, y_eval = y[:split], y[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "eval_metric": "auc",
        "seed": 42,
    }

    evals = [(dtrain, "train"), (deval, "eval")]
    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=10,
        evals=evals,
        evals_result=evals_result,
    )

    assert "train" in evals_result
    assert "auc" in evals_result["train"]
    assert "eval" in evals_result
    assert "auc" in evals_result["eval"]

    train_auc = evals_result["train"]["auc"]
    eval_auc = evals_result["eval"]["auc"]

    # AUC should be in [0, 1]
    for v in train_auc + eval_auc:
        assert 0.0 <= v <= 1.0 or np.isnan(v), f"AUC out of range: {v}"

    # Compare with sklearn macro-average for sanity check
    preds = bst.predict(deval)
    sklearn_auc = roc_auc_score(y_eval, preds, average="macro")
    builtin_auc = eval_auc[-1]

    # They won't be identical due to different approximations, but should be close
    assert abs(sklearn_auc - builtin_auc) < 0.15, (
        f"Built-in AUC {builtin_auc} too far from sklearn {sklearn_auc}"
    )

    print(f"Built-in multilabel AUC: {builtin_auc}")
    print(f"Sklearn macro AUC:       {sklearn_auc}")


def test_multilabel_aucpr_cpu():
    """Test that built-in aucpr works for multi-output binary classification."""
    n_samples = 256
    n_features = 16
    n_targets = 3

    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_targets,
        n_labels=2,
        random_state=42,
    )

    split = int(n_samples * 0.8)
    X_train, X_eval = X[:split], X[split:]
    y_train, y_eval = y[:split], y[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "eval_metric": "aucpr",
        "seed": 42,
    }

    evals = [(deval, "eval")]
    evals_result = {}
    xgb.train(params, dtrain, num_boost_round=5, evals=evals, evals_result=evals_result)

    assert "eval" in evals_result
    assert "aucpr" in evals_result["eval"]
    for v in evals_result["eval"]["aucpr"]:
        assert 0.0 <= v <= 1.0 or np.isnan(v), f"AUC-PR out of range: {v}"


if __name__ == "__main__":
    test_multilabel_auc_cpu()
    test_multilabel_aucpr_cpu()
    print("All multilabel AUC smoke tests passed.")
