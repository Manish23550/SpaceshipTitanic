"""
main.py
=======
Spaceship Titanic — Complete ML Pipeline Orchestrator.

USAGE
-----
    # From the SpaceshipTitanic/ root directory:
    python main.py

    # Skip Optuna tuning (faster iteration):
    python main.py --skip-tuning

    # Tune only specific models:
    python main.py --tune LightGBM XGBoost

WHAT THIS DOES
--------------
1.  Load raw train.csv and test.csv from data/
2.  Engineer features (50+ new columns)
3.  Build preprocessing pipelines (tree + linear + CatBoost)
4.  Train 7 base models with 5-fold CV
5.  Print and plot comprehensive evaluation metrics
6.  Tune LightGBM, XGBoost, and CatBoost with Optuna (100 trials each)
7.  Re-train tuned models and collect OOF predictions
8.  Build 4 ensemble strategies and compare them
9.  Generate and validate the final submission CSV
10. Save all models, figures, and logs to outputs/

LEADERBOARD EXPECTATION
-----------------------
With tuning + ensembling, this pipeline typically reaches ~0.81 accuracy
on the public leaderboard.  Top solutions achieve ~0.83 by using neural
networks, more creative feature engineering, or pseudo-labelling.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    CATBOOST_PARAMS,
    LGBM_PARAMS,
    MODELS_DIR,
    N_FOLDS,
    RANDOM_SEED,
    SAMPLE_SUB_PATH,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    XGB_PARAMS,
)
from utils import ensure_dirs, load_data, set_global_seed, setup_logger, timer
from feature_engineering import engineer_features
from preprocess import (
    build_linear_preprocessor,
    build_tree_preprocessor,
    get_feature_columns,
    prepare_for_catboost,
    preprocess,
)
from models import (
    build_soft_voting_ensemble,
    build_stacking_ensemble,
    find_optimal_blend_weights,
    get_all_models,
    save_model,
    train_all_models,
    train_cv,
    tune_model,
    weighted_average_ensemble,
)
from evaluation import (
    evaluate_oof,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curves,
    plot_model_comparison,
    plot_roc_curves,
    print_metrics_table,
)

log = setup_logger("main")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spaceship Titanic ML Pipeline"
    )
    parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Skip Optuna hyperparameter tuning (use config defaults).",
    )
    parser.add_argument(
        "--tune", nargs="+", default=["LightGBM", "XGBoost", "CatBoost"],
        help="Which models to tune (default: LightGBM XGBoost CatBoost).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Number of Optuna trials per model (default: 100).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def step_load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """STEP 1 — Load raw CSVs."""
    log.info("═══ STEP 1: Data Loading ═══")
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    log.info("Train shape: %s | Test shape: %s", train.shape, test.shape)
    return train, test


def step_feature_engineering(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """STEP 2 — Feature engineering."""
    log.info("═══ STEP 2: Feature Engineering ═══")
    with timer("Train feature engineering"):
        train_fe = engineer_features(train, is_train=True)
    with timer("Test feature engineering"):
        test_fe  = engineer_features(test, is_train=False)
    log.info("Train FE shape: %s | Test FE shape: %s", train_fe.shape, test_fe.shape)
    return train_fe, test_fe


def step_preprocessing(
    train_fe: pd.DataFrame,
    test_fe:  pd.DataFrame,
) -> tuple:
    """
    STEP 3 — Build and fit preprocessing pipelines.
    Returns arrays ready for each model type.
    """
    log.info("═══ STEP 3: Preprocessing ═══")

    # Identify feature columns
    X_train_raw, y_train, X_test_raw, num_cols, cat_cols = preprocess(
        train_fe, test_fe
    )

    # ── Tree preprocessor (no scaling) ──────────────────────────────────────
    with timer("Fit tree preprocessor"):
        tree_prep = build_tree_preprocessor(num_cols, cat_cols)
        X_train_tree = tree_prep.fit_transform(X_train_raw)
        X_test_tree  = tree_prep.transform(X_test_raw)
    log.info("Tree feature matrix: train=%s, test=%s", X_train_tree.shape, X_test_tree.shape)

    # ── Linear preprocessor (with scaling) ──────────────────────────────────
    with timer("Fit linear preprocessor"):
        linear_prep = build_linear_preprocessor(num_cols, cat_cols)
        X_train_linear = linear_prep.fit_transform(X_train_raw)
        X_test_linear  = linear_prep.transform(X_test_raw)

    # ── CatBoost preparation ─────────────────────────────────────────────────
    with timer("Prepare CatBoost data"):
        X_train_cb, cat_feat_indices = prepare_for_catboost(
            X_train_raw, num_cols, cat_cols
        )
        X_test_cb, _ = prepare_for_catboost(X_test_raw, num_cols, cat_cols)
        X_train_cb   = X_train_cb.values
        X_test_cb    = X_test_cb.values

    y = y_train.values

    # All feature names for importance plots (tree preprocessor output)
    feature_names = num_cols + cat_cols

    return (
        X_train_tree, X_test_tree,
        X_train_linear, X_test_linear,
        X_train_cb, X_test_cb,
        y, cat_feat_indices, feature_names,
    )


def step_train_base_models(
    X_tree: np.ndarray,
    X_linear: np.ndarray,
    X_catboost: np.ndarray,
    y: np.ndarray,
    cat_features: list[int],
) -> dict:
    """STEP 4 — Train all 7 base models with Stratified 5-Fold CV."""
    log.info("═══ STEP 4: Training Base Models ═══")
    with timer("Train all base models"):
        results = train_all_models(
            X_tree, X_linear, X_catboost, y, cat_features
        )
    return results


def step_evaluate(
    results: dict,
    y: np.ndarray,
    X_tree: np.ndarray,
    feature_names: list[str],
) -> None:
    """STEP 7 — Full evaluation: metrics table, ROC curves, feature importances."""
    log.info("═══ STEP 7: Evaluation ═══")

    # Metrics table
    metrics_df = print_metrics_table(results, y)

    # ROC curves for all models
    oof_probas = {n: r["oof_proba"] for n, r in results.items()}
    plot_roc_curves(y, oof_probas)

    # Confusion matrix for best model
    best_model_name = metrics_df.iloc[0]["Model"]
    plot_confusion_matrix(y, results[best_model_name]["oof_proba"], best_model_name)

    # Feature importance for tree models
    tree_model_names = [
        n for n in results if n not in ("LogisticRegression", "CatBoost")
    ]
    for name in tree_model_names[:3]:   # plot top 3 to avoid too many files
        fold_model = results[name]["fold_models"][0]   # first fold model
        plot_feature_importance(fold_model, feature_names, name, top_n=25)

    # Learning curve for best model
    from sklearn.base import clone
    from lightgbm import LGBMClassifier
    best_est = clone(results[best_model_name]["fold_models"][0])
    try:
        plot_learning_curves(best_est, X_tree, y, best_model_name)
    except Exception as e:
        log.warning("Could not plot learning curves for %s: %s", best_model_name, e)


def step_tune(
    args: argparse.Namespace,
    X_tree: np.ndarray,
    X_catboost: np.ndarray,
    y: np.ndarray,
    cat_features: list[int],
) -> dict:
    """STEP 5 — Optuna hyperparameter tuning."""
    if args.skip_tuning:
        log.info("═══ STEP 5: Tuning SKIPPED (--skip-tuning) ═══")
        return {}

    log.info("═══ STEP 5: Hyperparameter Tuning (%d trials each) ═══", args.n_trials)
    best_params = {}

    for model_name in args.tune:
        log.info("Tuning %s …", model_name)
        with timer(f"Optuna tuning: {model_name}"):
            params = tune_model(
                model_name,
                X_tree, X_catboost, y, cat_features,
                n_trials=args.n_trials,
            )
        best_params[model_name] = params

    return best_params


def step_ensemble(
    results: dict,
    X_tree: np.ndarray,
    X_test_tree: np.ndarray,
    y: np.ndarray,
    best_params: dict,
) -> tuple[np.ndarray, dict]:
    """STEP 6 — Build and compare ensemble strategies."""
    log.info("═══ STEP 6: Ensembling ═══")

    # Gather OOF probabilities from individual models
    oof_probas  = {n: r["oof_proba"] for n, r in results.items()}

    # Gather test predictions from fold models (average over folds)
    # CatBoost must predict on its own float-free matrix; all others use X_test_tree.
    test_probas = {}
    for name, res in results.items():
        fold_preds = []
        X_pred = X_test_cb if name == "CatBoost" else (
            X_test_linear if name == "LogisticRegression" else X_test_tree
        )
        for fold_model in res["fold_models"]:
            fold_preds.append(fold_model.predict_proba(X_pred)[:, 1])
        test_probas[name] = np.mean(fold_preds, axis=0)

    ensemble_scores = {}

    # ── Strategy 1: Soft Voting ──────────────────────────────────────────────
    with timer("Soft voting ensemble"):
        lgbm_p = best_params.get("LightGBM", None)
        xgb_p  = best_params.get("XGBoost",  None)
        voting_model, voting_acc = build_soft_voting_ensemble(
            X_tree, y, lgbm_p, xgb_p
        )
    test_probas["SoftVoting"] = voting_model.predict_proba(X_test_tree)[:, 1]
    # OOF for voting via CV predict
    from sklearn.model_selection import cross_val_predict
    voting_oof = cross_val_predict(
        voting_model, X_tree, y, cv=N_FOLDS, method="predict_proba", n_jobs=1
    )[:, 1]
    oof_probas["SoftVoting"] = voting_oof
    ensemble_scores["SoftVoting"] = voting_acc
    save_model(voting_model, "SoftVotingEnsemble")

    # ── Strategy 2: Stacking ─────────────────────────────────────────────────
    with timer("Stacking ensemble"):
        stacking_model, stacking_acc = build_stacking_ensemble(X_tree, y)
    test_probas["Stacking"] = stacking_model.predict_proba(X_test_tree)[:, 1]
    stacking_oof = cross_val_predict(
        stacking_model, X_tree, y, cv=N_FOLDS, method="predict_proba", n_jobs=1
    )[:, 1]
    oof_probas["Stacking"] = stacking_oof
    ensemble_scores["Stacking"] = stacking_acc
    save_model(stacking_model, "StackingEnsemble")

    # ── Strategy 3: Weighted Average ─────────────────────────────────────────
    # Use only the 4 best individual models by OOF accuracy
    from sklearn.metrics import accuracy_score
    model_accs = {
        n: accuracy_score(y, (oof_probas[n] >= 0.5).astype(int))
        for n in list(results.keys())
    }
    top4 = sorted(model_accs, key=model_accs.get, reverse=True)[:4]
    oof4  = {n: oof_probas[n]  for n in top4}
    test4 = {n: test_probas[n] for n in top4}

    weighted_test, weighted_acc = weighted_average_ensemble(oof4, test4, y)
    test_probas["WeightedAvg"] = weighted_test
    ensemble_scores["WeightedAvg"] = weighted_acc
    log.info("Weighted Average ensemble OOF accuracy: %.4f", weighted_acc)

    # ── Strategy 4: Probability Blending (optimised weights) ─────────────────
    blend_test, blend_acc, blend_weights = find_optimal_blend_weights(
        oof4, test4, y, n_search=500
    )
    test_probas["ProbBlending"] = blend_test
    ensemble_scores["ProbBlending"] = blend_acc

    # Compare all ensembles
    plot_model_comparison(results, ensemble_scores)

    # Best ensemble test proba = highest OOF accuracy
    best_ensemble = max(ensemble_scores, key=ensemble_scores.get)
    log.info(
        "Best ensemble strategy: %s (OOF acc=%.4f)",
        best_ensemble, ensemble_scores[best_ensemble],
    )

    return test_probas[best_ensemble], ensemble_scores


def step_submission(
    test_df: pd.DataFrame,
    test_proba: np.ndarray,
) -> Path:
    """STEP 8 — Generate, validate, and save submission CSV."""
    log.info("═══ STEP 8: Generating Submission ═══")
    from utils import validate_submission

    preds      = (test_proba >= 0.5)
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": preds,
    })

    # Validate before saving
    validate_submission(submission, SAMPLE_SUB_PATH)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "final_submission.csv"
    submission.to_csv(out_path, index=False)
    log.info("✅ Submission saved: %s  (%d rows)", out_path, len(submission))
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    set_global_seed(RANDOM_SEED)
    ensure_dirs()
    log.info("══════════════════════════════════════════════")
    log.info("  Spaceship Titanic ML Pipeline")
    log.info("  Random seed : %d", RANDOM_SEED)
    log.info("  CV folds    : %d", N_FOLDS)
    log.info("══════════════════════════════════════════════")

    # ── Steps ────────────────────────────────────────────────────────────────
    train, test = step_load_data()

    train_fe, test_fe = step_feature_engineering(train, test)

    (
        X_train_tree, X_test_tree,
        X_train_linear, X_test_linear,
        X_train_cb, X_test_cb,
        y, cat_feat_indices, feature_names,
    ) = step_preprocessing(train_fe, test_fe)

    results = step_train_base_models(
        X_train_tree, X_train_linear, X_train_cb, y, cat_feat_indices
    )

    best_params = step_tune(
        args, X_train_tree, X_train_cb, y, cat_feat_indices
    )

    step_evaluate(results, y, X_train_tree, feature_names)

    best_test_proba, ensemble_scores = step_ensemble(
        results, X_train_tree, X_test_tree, y, best_params
    )

    sub_path = step_submission(test, best_test_proba)

    log.info("═══ Pipeline Complete ═══")
    log.info("Submission: %s", sub_path)


if __name__ == "__main__":
    main()
