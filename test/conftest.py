import pytest


@pytest.fixture(scope="module")
def test_data_dir(request):
    """Return the path of test data directory."""
    return str(request.fspath.join("..", "..", "data"))
