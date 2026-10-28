"""Cloud startup without model downloads or provider requests."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import deployment


def test_runtime_uses_app_directory_and_preserves_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv('LLM_MODEL', 'existing')
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    deployment.configure_runtime(tmp_path, {'LLM_MODEL': 'other', 'LLM_API_KEY': 'test-only', 'unrelated': 'x'})
    assert deployment.Path.cwd() == tmp_path
    assert deployment.os.environ['LLM_MODEL'] == 'existing'
    assert deployment.os.environ['LLM_API_KEY'] == 'test-only'
    monkeypatch.delenv('LLM_API_KEY')


@pytest.mark.parametrize('state', ['missing', 'empty', 'populated', 'broken', 'zero'])
def test_index_initialization(tmp_path, monkeypatch, state):
    class NotFoundError(Exception):
        pass
    client = Mock()
    if state == 'missing':
        client.get_collection.side_effect = NotFoundError()
    elif state == 'broken':
        client.get_collection.side_effect = PermissionError('storage unavailable')
    else:
        client.get_collection.return_value.count.return_value = 108 if state == 'populated' else 0
    persistent = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, 'chromadb', SimpleNamespace(PersistentClient=persistent))
    monkeypatch.setitem(sys.modules, 'chromadb.errors', SimpleNamespace(NotFoundError=NotFoundError))
    import build_index
    builder = Mock(return_value=0 if state == 'zero' else 108)
    monkeypatch.setattr(build_index, 'build_index', builder)
    if state in ('broken', 'zero'):
        with pytest.raises(PermissionError if state == 'broken' else RuntimeError):
            deployment.ensure_index(tmp_path)
    else:
        assert deployment.ensure_index(tmp_path) == 108
    persistent.assert_called_once_with(path=str(tmp_path / 'data' / 'index'))
    if state in ('populated', 'broken'):
        builder.assert_not_called()
    else:
        builder.assert_called_once_with(tmp_path / 'data' / 'cards.jsonl', tmp_path / 'data' / 'index', client=client)
