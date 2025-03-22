from typing import Optional, Dict, Any, Set
from collections import defaultdict
from utils.logging_util import logger
import hashlib
from datetime import datetime
import logging
from Levenshtein import distance
import json
import os

class MockRedis:
    """Mock Redis implementation for local debugging"""
    
    def __init__(self):
        self.data = {}  # Key-value store
        self.sets = defaultdict(set)  # Set store
        self.hashes = defaultdict(dict)  # Hash store
    
    def exists(self, key: str) -> bool:
        return key in self.data or key in self.sets or key in self.hashes
    
    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)
    
    def set(self, key: str, value: str) -> bool:
        self.data[key] = value
        return True
    
    def smembers(self, key: str) -> Set[str]:
        return self.sets.get(key, set())
    
    def sadd(self, key: str, value: str) -> bool:
        if key not in self.sets:
            self.sets[key] = set()
        self.sets[key].add(value)
        return True
    
    def hget(self, key: str, field: str) -> Optional[str]:
        return self.hashes.get(key, {}).get(field)
    
    def hset(self, key: str, field: str, value: str) -> bool:
        if key not in self.hashes:
            self.hashes[key] = {}
        self.hashes[key][field] = value
        return True
    
    def ping(self) -> bool:
        return True

class MockRedisQueryTool:
    """Mock Redis query tool for testing"""
    
    def __init__(self):
        self.redis = MockRedis()
        self.question_key_prefix = "qa:question:"
        self.index_key = "qa:index"
        self.similarity_threshold = 0.6  # Lower threshold for more lenient matching
        self._load_qa_pairs()
    
    def _hash_question(self, question: str) -> str:
        """Generate a hash for the question"""
        return hashlib.md5(question.encode()).hexdigest()
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance and word overlap"""
        if not str1 or not str2:
            return 0.0
        
        # Convert to lowercase for case-insensitive comparison
        str1 = str1.lower()
        str2 = str2.lower()
        
        # Calculate Levenshtein distance similarity
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        levenshtein_sim = 1 - (distance(str1, str2) / max_len)
        
        # Calculate word overlap similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Remove common words for better matching
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        word_sim = overlap / total if total > 0 else 0.0
        
        # Combine both similarities with adjusted weights
        return 0.4 * levenshtein_sim + 0.6 * word_sim
    
    def _fuzzy_search(self, question: str) -> Optional[Dict[str, Any]]:
        """Find similar questions using string similarity"""
        try:
            best_match = None
            best_similarity = 0
            
            # Get all question keys from index
            question_keys = self.redis.smembers(self.index_key)
            
            for key in question_keys:
                stored_question = self.redis.hget(key, "question")
                if stored_question:
                    similarity = self._calculate_similarity(question, stored_question)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = key
            
            if best_match and best_similarity >= self.similarity_threshold:
                return {
                    "answer": self.redis.hget(best_match, "answer"),
                    "category": self.redis.hget(best_match, "category"),
                    "similarity": best_similarity
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Error in fuzzy search: {str(e)}")
            return None
    
    def query(self, question: str) -> Optional[Dict[str, Any]]:
        """Query the mock Redis for an answer"""
        if not question or not isinstance(question, str) or len(question) > 500:
            return None
            
        try:
            # Try exact match first
            question_hash = self._hash_question(question)
            key = f"{self.question_key_prefix}{question_hash}"
            
            if self.redis.exists(key):
                return {
                    "answer": self.redis.hget(key, "answer"),
                    "category": self.redis.hget(key, "category"),
                    "similarity": 1.0
                }
            
            # Try fuzzy search
            return self._fuzzy_search(question)
            
        except Exception as e:
            logging.error(f"Error in query: {str(e)}")
            return None
    
    def _add_qa_pair(self, qa_pair: Dict[str, str]) -> None:
        """Add a Q&A pair to the mock Redis"""
        try:
            question_hash = self._hash_question(qa_pair["question"])
            key = f"{self.question_key_prefix}{question_hash}"
            
            # Store Q&A pair
            self.redis.hset(key, "question", qa_pair["question"])
            self.redis.hset(key, "answer", qa_pair["answer"])
            self.redis.hset(key, "category", qa_pair["category"])
            
            # Add to index
            self.redis.sadd(self.index_key, key)
            
        except Exception as e:
            logging.error(f"Error adding Q&A pair: {str(e)}")
    
    def _load_qa_pairs(self) -> None:
        """Load Q&A pairs from qa_pairs.json file"""
        try:
            # Get the absolute path to qa_pairs.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            qa_pairs_path = os.path.join(os.path.dirname(current_dir), 'data', 'qa_pairs.json')
            
            # Read and parse the JSON file
            with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add each Q&A pair to Redis
            for qa_pair in data.get('qa_pairs', []):
                self._add_qa_pair(qa_pair)
                
        except Exception as e:
            logging.error(f"Error loading Q&A pairs: {str(e)}")
            raise 