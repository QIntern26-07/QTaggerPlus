from classical.__main__ import aggregate_records


def test_aggregate_records_computes_mean_and_std():
    records = [
        {"model": "random_forest", "task": "binary",
         "metrics": {"accuracy": 0.90, "f1_macro": 0.88}},
        {"model": "random_forest", "task": "binary",
         "metrics": {"accuracy": 0.94, "f1_macro": 0.92}},
    ]
    agg = aggregate_records(records)
    assert abs(agg["accuracy_mean"] - 0.92) < 1e-9
    assert abs(agg["f1_macro_mean"] - 0.90) < 1e-9
    assert agg["accuracy_std"] > 0
