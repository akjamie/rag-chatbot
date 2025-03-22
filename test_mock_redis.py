import pytest
from tests.mock_redis import MockVectorSearchTool
import numpy as np

def test_vector_search_exact_match():
    """Test exact match functionality of mock vector search tool"""
    # Initialize the tool
    tool = MockVectorSearchTool()
    
    # Test exact match
    result = tool.query("IDAM")
    assert result is not None
    assert result["answer"] == "IDAM is identity and access management system"
    assert result["category"] == "geography"
    assert result["similarity"] > 0.9  # High similarity for exact match
    
    # Test another exact match
    result = tool.query("ITOP")
    assert result is not None
    assert result["answer"] == "ITOP is information technology operation platform"
    assert result["category"] == "technology"
    assert result["similarity"] > 0.9

def test_vector_search_semantic_match():
    """Test semantic matching functionality using vector similarity"""
    tool = MockVectorSearchTool()
    
    # Test semantic match with related but different wording
    result = tool.query("What is the identity management system?")
    assert result is not None
    assert "IDAM" in result["answer"]
    assert result["similarity"] > 0.7  # Good similarity for semantic match
    
    # Test semantic match with abbreviation expansion
    result = tool.query("What does ITOP stand for?")
    assert result is not None
    assert "information technology operation platform" in result["answer"].lower()
    assert result["similarity"] > 0.7

def test_vector_search_no_match():
    """Test cases where no match should be found"""
    tool = MockVectorSearchTool()
    
    # Test completely unrelated query
    result = tool.query("What is the weather forecast for tomorrow?")
    assert result is None
    
    # Test empty query
    result = tool.query("")
    assert result is None
    
    # Test query with just common words
    result = tool.query("the and with")
    assert result is None

def test_vector_search_threshold_impact():
    """Test the impact of different similarity thresholds"""
    tool = MockVectorSearchTool()
    
    # Test with default threshold
    default_result = tool.query("information management systems")
    
    # Test with higher threshold
    tool.similarity_threshold = 0.9
    high_threshold_result = tool.query("information management systems")
    
    # Test with lower threshold
    tool.similarity_threshold = 0.5
    low_threshold_result = tool.query("information management systems")
    
    # Lower threshold should find more matches
    assert (high_threshold_result is None) or (default_result is not None and high_threshold_result is not None)
    assert low_threshold_result is not None
    
    # Reset threshold for other tests
    tool.similarity_threshold = 0.7

def test_vector_search_custom_embeddings():
    """Test adding custom embeddings and querying for them"""
    tool = MockVectorSearchTool()
    
    # Add a new Q&A pair with custom embedding
    custom_embedding = np.random.rand(384).astype(np.float32)  # Common embedding size
    tool.add_qa_with_embedding(
        question="What is vector search?",
        answer="Vector search uses semantic embeddings to find similar content based on meaning rather than exact text matches.",
        category="technology",
        embedding=custom_embedding
    )
    
    # Test querying for the new entry
    result = tool.query("Tell me about vector-based search")
    assert result is not None
    assert "semantic embeddings" in result["answer"]
    assert result["category"] == "technology"

def test_vector_search_error_handling():
    """Test error handling in the vector search tool"""
    tool = MockVectorSearchTool()
    
    # Test with None input
    result = tool.query(None)
    assert result is None
    
    # Test with non-string input
    result = tool.query(123)
    assert result is None
    
    # Test with very long string
    result = tool.query("a" * 1000)
    assert result is None
    
    # Test with malformed embedding
    with pytest.raises(ValueError):
        tool.add_qa_with_embedding(
            question="Bad embedding",
            answer="This should fail",
            category="test",
            embedding=np.array([1, 2, 3])  # Wrong shape
        ) 