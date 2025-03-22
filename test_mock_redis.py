import pytest
from tests.mock_redis import MockRedisQueryTool

def test_mock_redis_query_tool():
    """Test basic functionality of mock Redis query tool"""
    # Initialize the tool
    tool = MockRedisQueryTool()
    
    # Test exact match
    result = tool.query("IDAM")
    assert result is not None
    assert result["answer"] == "IDAM is identity and access management system"
    assert result["category"] == "geography"
    
    # Test fuzzy match with slight variation
    result = tool.query("ITOP")
    assert result is not None
    assert result["answer"] == "ITOP is information technology operation platform"
    
    # Test no match
    result = tool.query("IMAX")
    assert result is None
    
    # Test empty question
    result = tool.query("")
    assert result is None
    
    # Test unrelated question
    result = tool.query("What is the weather like?")
    assert result is None



def test_mock_redis_query_tool_similarity_threshold():
    """Test the tool's behavior with different similarity thresholds"""
    tool = MockRedisQueryTool()
    
    # Test with default threshold (0.5)
    tool.similarity_threshold = 0.5
    result = tool.query("idam")
    assert result is not None

    
    # Test with lower threshold (0.5)
    tool.similarity_threshold = 0.5
    result = tool.query("itop")
    assert result is not None

    
    # Test with higher threshold (0.9)
    tool.similarity_threshold = 0.9
    result = tool.query("imax")
    assert result is None

def test_mock_redis_query_tool_error_handling():
    """Test error handling in the tool"""
    tool = MockRedisQueryTool()
    
    # Test with None input
    result = tool.query(None)
    assert result is None
    
    # Test with non-string input
    result = tool.query(123)
    assert result is None
    
    # Test with very long string
    result = tool.query("a" * 1000)
    assert result is None 