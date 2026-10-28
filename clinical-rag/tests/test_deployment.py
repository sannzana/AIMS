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


def test_default_index_is_independent_of_working_directory(tmp_path, monkeypatch):
    import retrieve
    monkeypatch.chdir(tmp_path)
    expected = deployment.Path(retrieve.__file__).resolve().parents[1] / 'data' / 'index'
    assert retrieve._index_path(deployment.Path('data/index')) == expected
    assert retrieve._index_path(expected) == expected


def test_real_chroma_startup_collection_opens_through_retriever(tmp_path, monkeypatch):
    # Real persistence/client handles, synthetic vectors: no network or LLM.
    import shutil
    import retrieve
    from build_index import load_cards
    source = deployment.Path(__file__).resolve().parents[1] / 'data' / 'cards.jsonl'
    (tmp_path / 'data').mkdir()
    shutil.copyfile(source, tmp_path / 'data' / 'cards.jsonl')
    class Embedder:
        def __init__(self, name): pass
        def encode(self, texts):
            return [[1.0, 0.5, 0.1] for _ in texts]
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Embedder))
    assert deployment.ensure_index(tmp_path) == 108
    monkeypatch.chdir(tmp_path)
    retrieve._resources.cache_clear()
    relative = deployment.Path('data') / '..' / 'data' / 'index'
    cards = {c.card_id: c for c in load_cards(source)}
    engine = retrieve.Retriever(relative, cards=cards)
    assert engine.collection.count() == 108
    absolute = retrieve.Retriever(tmp_path / 'data' / 'index', cards=cards)
    assert absolute.collection.id == engine.collection.id
    retrieve._resources.cache_clear()
