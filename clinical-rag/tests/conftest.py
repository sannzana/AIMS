from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


import pytest


@pytest.fixture(autouse=True)
def isolated_ui_startup(request, monkeypatch):
    # Existing UI tests inject retrieval/model results; they must not download
    # an embedding model or build a real index as a side effect of rendering.
    # The startup module itself is exercised separately in test_deployment.py.
    if request.module.__name__.endswith("test_deployment"):
        return
    import deployment
    monkeypatch.setattr(deployment, "ensure_index", lambda root: 108)
