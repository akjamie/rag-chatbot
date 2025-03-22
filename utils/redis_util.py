import json
import os
from typing import Dict, Any, List
import redis
from datetime import datetime
from utils.logging_util import logger

class RedisUtil:
    """Utility class for managing Redis Q&A data"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.question_key_prefix = "qa:question:"
        self.index_key = "qa:index:questions"
        self.category_key_prefix = "qa:category:"
        self.expire_time = 86400  # 24 hours in seconds
    
    def import_from_json(self, file_path: str) -> int:
        """Import Q&A pairs from a JSON file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"JSON file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'qa_pairs' not in data:
                raise ValueError("Invalid JSON format: must contain 'qa_pairs' array")
            
            qa_pairs = data['qa_pairs']
            imported_count = 0
            
            for qa_pair in qa_pairs:
                if not isinstance(qa_pair, dict) or 'question' not in qa_pair or 'answer' not in qa_pair:
                    logger.warning(f"Skipping invalid Q&A pair: {qa_pair}")
                    continue
                
                # Save Q&A pair
                self.save_qa(
                    question=qa_pair['question'],
                    answer=qa_pair['answer'],
                    category=qa_pair.get('category', 'general'),
                    created_at=qa_pair.get('created_at', datetime.now().isoformat())
                )
                imported_count += 1
            
            logger.info(f"Successfully imported {imported_count} Q&A pairs from {file_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing from JSON: {str(e)}")
            raise
    
    def save_qa(self, question: str, answer: str, category: str = 'general', created_at: str = None):
        """Save a Q&A pair to Redis"""
        try:
            # Calculate hash of question as key
            question_hash = self._hash_question(question)
            key = f"{self.question_key_prefix}{question_hash}"
            
            # Store QA pair with metadata
            self.redis.hmset(key, {
                "question": question,
                "answer": answer,
                "category": category,
                "created_at": created_at or datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            # Set expiration time
            self.redis.expire(key, self.expire_time)
            
            # Add to index
            self.redis.sadd(self.index_key, question_hash)
            
            # Add to category index
            self.redis.sadd(f"{self.category_key_prefix}{category}", question_hash)
            
            logger.debug(f"Saved Q&A pair with hash: {question_hash}")
            
        except Exception as e:
            logger.error(f"Error saving Q&A to Redis: {str(e)}")
            raise
    
    def get_qa_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all Q&A pairs for a specific category"""
        try:
            question_hashes = self.redis.smembers(f"{self.category_key_prefix}{category}")
            qa_pairs = []
            
            for q_hash in question_hashes:
                key = f"{self.question_key_prefix}{q_hash.decode()}"
                qa_pair = self.redis.hgetall(key)
                if qa_pair:
                    qa_pairs.append({
                        "question": qa_pair[b"question"].decode(),
                        "answer": qa_pair[b"answer"].decode(),
                        "category": qa_pair[b"category"].decode(),
                        "created_at": qa_pair[b"created_at"].decode()
                    })
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error getting Q&A pairs by category: {str(e)}")
            return []
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        try:
            # Get all keys matching the category prefix
            category_keys = self.redis.keys(f"{self.category_key_prefix}*")
            return [key.decode().replace(self.category_key_prefix, '') for key in category_keys]
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []
    
    def clear_all_qa(self):
        """Clear all Q&A data from Redis"""
        try:
            # Get all keys to delete
            question_keys = self.redis.keys(f"{self.question_key_prefix}*")
            category_keys = self.redis.keys(f"{self.category_key_prefix}*")
            
            # Delete all keys
            if question_keys:
                self.redis.delete(*question_keys)
            if category_keys:
                self.redis.delete(*category_keys)
            
            # Delete index
            self.redis.delete(self.index_key)
            
            logger.info("Successfully cleared all Q&A data")
            
        except Exception as e:
            logger.error(f"Error clearing Q&A data: {str(e)}")
            raise
    
    @staticmethod
    def _hash_question(question: str) -> str:
        """Generate a hash for the question"""
        import hashlib
        return hashlib.md5(question.encode()).hexdigest() 