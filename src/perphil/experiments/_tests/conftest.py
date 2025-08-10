def pytest_configure(config):
    # Mark used by regression tests in this repo
    config.addinivalue_line(
        "markers", "regression: marks regression tests using pytest-regressions"
    )
