"""Verify that dense embedding model loading falls back to the configured
fallback model when the primary model can't be loaded, instead of crashing
the whole pipeline (spec: "handle missing evidence gracefully").
"""
import backend.retrieval.dense as dense_module


def test_get_embedding_model_falls_back_on_primary_failure(monkeypatch):
    dense_module._model_cache.clear()
    calls = []

    class _FakeModel:
        def __init__(self, name):
            calls.append(name)
            if name == "primary-that-does-not-exist":
                raise OSError("simulated: model not found")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeModel)

    model, actual_name = dense_module.get_embedding_model("primary-that-does-not-exist")
    assert actual_name == dense_module.settings.dense_embedding_fallback_model
    assert calls == ["primary-that-does-not-exist", dense_module.settings.dense_embedding_fallback_model]


def test_get_embedding_model_uses_cache_on_second_call(monkeypatch):
    dense_module._model_cache.clear()
    calls = []

    class _FakeModel:
        def __init__(self, name):
            calls.append(name)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeModel)

    dense_module.get_embedding_model("some-model")
    dense_module.get_embedding_model("some-model")
    assert calls == ["some-model"]  # second call hit the cache, no re-instantiation
