import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--enable-example-tests", action="store_true", default=False,
        help="Also test the examples, which may be very slow")

def pytest_configure(config):
    config.addinivalue_line("markers", "example: mark test as an example test")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--enable-example-tests"):
        return

    skip_example = pytest.mark.skip(reason="Need --enable-example-tests option to run")
    for item in items:
        if "example" in item.keywords:
            item.add_marker(skip_example)
