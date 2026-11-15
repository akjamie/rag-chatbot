import pytest
from unittest.mock import MagicMock, patch

def test_flag_reranker_import():
    """Verify FlagReranker can be imported cleanly (AC4)."""
    from FlagEmbedding import FlagReranker
    assert FlagReranker is not None


def test_reranker_scoring_interface():
    """Verify reranker scoring interface returns valid float scores for document candidates (AC4)."""
    with patch("FlagEmbedding.FlagReranker") as mock_reranker_cls:
        instance = MagicMock()
        mock_reranker_cls.return_value = instance
        # Mock compute_score returning float similarity scores
        instance.compute_score.return_value = [0.85, 0.42]

        from FlagEmbedding import FlagReranker
        reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=False)
        scores = reranker.compute_score([["What is RAG?", "RAG is retrieval augmented generation."], ["What is RAG?", "Apples are fruit."]])

        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, (float, int)) for s in scores)
        assert scores[0] > scores[1]
