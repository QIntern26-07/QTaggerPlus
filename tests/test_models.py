import optuna
import pytest
from classical import models


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_make_model_returns_fittable_estimator(name):
    est = models.make_model(name, params={}, task="binary")
    assert hasattr(est, "fit") and hasattr(est, "predict")


def test_svm_search_space_has_gamma_float_and_class_weight():
    study = optuna.create_study()
    captured = {}

    def objective(trial):
        captured.update(models.suggest_params("svm", trial))
        return 0.0

    study.optimize(objective, n_trials=1)
    assert "gamma" in captured and isinstance(captured["gamma"], float)
    assert "class_weight" in captured


def test_svm_model_has_no_predict_proba():
    m = models.make_model("svm", {"C": 1.0, "gamma": 0.1}, "binary")
    # probability estimation defaults to off in sklearn's SVC; predict_proba
    # stays hidden via available_if without needing to pass probability=False
    # explicitly (see models.make_model's comment on the FutureWarning this avoids)
    assert not hasattr(m, "predict_proba")
    assert m.decision_function_shape == "ovr"


def test_xgboost_objective_by_task():
    est_binary = models.make_model("xgboost", params={}, task="binary")
    assert est_binary.get_params()["objective"] == "binary:logistic"

    est_multiclass = models.make_model("xgboost", params={}, task="multiclass")
    assert est_multiclass.get_params()["objective"] == "multi:softprob"


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_suggest_params_returns_nonempty_dict(name):
    study = optuna.create_study()
    trial = study.ask()
    params = models.suggest_params(name, trial)
    assert isinstance(params, dict) and len(params) > 0


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_suggest_params_integrates_with_make_model(name):
    study = optuna.create_study()
    trial = study.ask()
    params = models.suggest_params(name, trial)
    est = models.make_model(name, params=params, task="binary")
    assert hasattr(est, "fit")


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "lightgbm"])
def test_make_model_threads_n_jobs(name):
    est = models.make_model(name, params={}, task="binary", n_jobs=1)
    assert est.get_params()["n_jobs"] == 1


def test_make_model_svm_ignores_n_jobs_param():
    # SVC has no n_jobs constructor arg; passing n_jobs must not raise.
    est = models.make_model("svm", params={}, task="binary", n_jobs=1)
    assert "n_jobs" not in est.get_params()
