import json
import pytest

try:
    from perphil.experiments.iterative_bench import Approach
    from perphil.experiments.petsc_profiling import (
        run_perf_once,
        run_perf_sweep,
        save_perf_json,
        save_perf_csv,
    )
except Exception:  # pragma: no cover
    pytest.skip("Firedrake not available", allow_module_level=True)


def test_run_perf_once_smoketest(tmp_path):
    res = run_perf_once(2, 2, Approach.PLAIN_GMRES, repeats=1, eager=False, backend="events")
    d = res.to_dict()
    # basic structure checks
    assert d["nx"] == 2 and d["ny"] == 2
    assert "time_total" in d and d["time_total"] >= 0.0
    assert "flops_total" in d and d["flops_total"] >= 0.0


def test_run_perf_sweep_and_save(tmp_path):
    df = run_perf_sweep([2], [Approach.PLAIN_GMRES], repeats=1, eager=False, backend="events")
    json_path = tmp_path / "perf.json"
    csv_path = tmp_path / "perf.csv"
    save_perf_json(df, str(json_path))
    save_perf_csv(df, str(csv_path))
    assert json_path.exists() and csv_path.exists()
    # sanity check JSON file
    data = json.loads(json_path.read_text())
    assert isinstance(data, list) and data


@pytest.mark.regression
def test_perf_to_dict_regression(data_regression):
    res = run_perf_once(2, 2, Approach.PLAIN_GMRES, repeats=1, eager=False, backend="events")
    # prune variable fields
    d = res.to_dict()
    volatile = {k for k in d.keys() if k.startswith("time_") or k.startswith("flops_")}
    # mflops_* are derived from flops/time and can vary; memory peaks also vary
    volatile |= {k for k in d.keys() if k.startswith("mflops_")}
    volatile |= {
        "iterations",
        "residual",
        "time_total",
        "time_total_repeats",
        "mem_rss_peak_kb",
        "mem_rss_delta_kb",
    }
    for k in list(volatile):
        d.pop(k, None)
    # firedrake commit hash is environment-dependent
    if "metadata" in d and isinstance(d["metadata"], dict):
        d["metadata"].pop("firedrake_version", None)
    data_regression.check(d)
