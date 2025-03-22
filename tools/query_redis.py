from typing import Optional, Dict, Any
import redis
from Levenshtein import distance
from utils.logging_util import logger
from config.common_settings import CommonConfig

class RedisQueryTool:
    """Tool for querying Q&A pairs from Redis"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize Redis query tool"""
        if redis_client is None:
            config = CommonConfig()
            redis_client = redis.Redis(
                host=config.get_env_variable("REDIS_HOST", "localhost"),
                port=int(config.get_env_variable("REDIS_PORT", 6379)),
                db=int(config.get_env_variable("REDIS_DB", 0)),
                decode_responses=True
            )
        self.redis = redis_client
        self.question_key_prefix = "qa:question:"
        self.index_key = "qa:index:questions"
        self.similarity_threshold = 0.8  # 80% similarity threshold
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance"""
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        return 1 - (distance(str1, str2) / max_len)
    
    def _fuzzy_search(self, question: str) -> Optional[Dict[str, Any]]:
        """Search for similar questions using fuzzy matching"""
        try:
            # Get all question keys
            question_keys = self.redis.smembers(self.index_key)
            best_match = None
            best_similarity = 0
            
            for key in question_keys:
                stored_question = self.redis.hget(key, "question")
                if stored_question:
                    similarity = self._calculate_similarity(question.lower(), stored_question.lower())
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = key
            
            if best_similarity >= self.similarity_threshold:
                return {
                    "question": self.redis.hget(best_match, "question"),
                    "answer": self.redis.hget(best_match, "answer"),
                    "category": self.redis.hget(best_match, "category"),
                    "similarity": best_similarity
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy search: {str(e)}")
            return None
    
    def query(self, question: str) -> Optional[Dict[str, Any]]:
        """Query Redis for an answer to the given question"""
        try:
            # First try exact match
            question_hash = self._hash_question(question)
            key = f"{self.question_key_prefix}{question_hash}"
            
            if self.redis.exists(key):
                return {
                    "question": self.redis.hget(key, "question"),
                    "answer": self.redis.hget(key, "answer"),
                    "category": self.redis.hget(key, "category"),
                    "similarity": 1.0
                }
            
            # If no exact match, try fuzzy search
            return self._fuzzy_search(question)
            
        except Exception as e:
            logger.error(f"Error querying Redis: {str(e)}")
            return None
    
    @staticmethod
    def _hash_question(question: str) -> str:
        """Generate a hash for the question"""
        import hashlib
        return hashlib.md5(question.lower().encode()).hexdigest() 