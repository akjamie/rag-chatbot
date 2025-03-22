import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
import redis
import numpy as np
from datetime import datetime
from utils.logging_util import logger
from sentence_transformers import SentenceTransformer

class RedisUtil:
    """Utility class for managing Redis Vector Database for Q&A data"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.question_key_prefix = "qa:question:"
        self.index_key = "qa:index:questions"
        self.category_key_prefix = "qa:category:"
        self.vector_key_prefix = "qa:vector:"
        self.expire_time = 0  # 24 hours in seconds
        
        # Load the sentence transformer model for text embedding
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimension embeddings
        self.embedding_dim = 384
        
        # Create vector index if it doesn't exist
        self._create_vector_index()
    
    def _create_vector_index(self):
        """Create a vector index in Redis if it doesn't exist"""
        try:
            # Check if the index exists
            index_list = self.redis.execute_command("FT._LIST")
            if b"qa_vector_idx" not in index_list:
                # Create the index with vector search capabilities
                create_cmd = [
                    "FT.CREATE", "qa_vector_idx", "ON", "HASH", "PREFIX", "1", self.vector_key_prefix,
                    "SCHEMA", "question", "TEXT", "answer", "TEXT", "category", "TAG",
                    "embedding", "VECTOR", "FLAT", "6", "TYPE", "FLOAT32", 
                    "DIM", str(self.embedding_dim), "DISTANCE_METRIC", "COSINE"
                ]
                self.redis.execute_command(*create_cmd)
                logger.info("Created vector search index in Redis")
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
    
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
                
                # Generate embedding for the question
                embedding = self._generate_embedding(qa_pair['question'])
                
                # Save Q&A pair with vector embedding
                self.save_qa(
                    question=qa_pair['question'],
                    answer=qa_pair['answer'],
                    category=qa_pair.get('category', 'general'),
                    created_at=qa_pair.get('created_at', datetime.now().isoformat()),
                    embedding=embedding
                )
                imported_count += 1
            
            logger.info(f"Successfully imported {imported_count} Q&A pairs from {file_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing from JSON: {str(e)}")
            raise
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text using sentence transformer"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding
    
    def _vector_to_redis_format(self, vector: np.ndarray) -> bytes:
        """Convert numpy vector to Redis byte format"""
        return vector.astype(np.float32).tobytes()
    
    def save_qa(self, question: str, answer: str, category: str = 'general', 
                created_at: str = None, embedding: Optional[np.ndarray] = None):
        """Save a Q&A pair to Redis with vector embedding"""
        try:
            # Calculate hash of question as key
            question_hash = self._hash_question(question)
            key = f"{self.question_key_prefix}{question_hash}"
            vector_key = f"{self.vector_key_prefix}{question_hash}"
            
            # Generate embedding if not provided
            if embedding is None:
                embedding = self._generate_embedding(question)
            
            # Store QA pair with metadata
            self.redis.hmset(key, {
                "question": question,
                "answer": answer,
                "category": category,
                "created_at": created_at or datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            # Store vector embedding with the same metadata
            self.redis.hmset(vector_key, {
                "question": question,
                "answer": answer,
                "category": category,
                "created_at": created_at or datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "embedding": self._vector_to_redis_format(embedding)
            })
            
            # Set expiration time
            self.redis.expire(key, self.expire_time)
            self.redis.expire(vector_key, self.expire_time)
            
            # Add to index
            self.redis.sadd(self.index_key, question_hash)
            
            # Add to category index
            self.redis.sadd(f"{self.category_key_prefix}{category}", question_hash)
            
            logger.debug(f"Saved Q&A pair with hash: {question_hash} and vector embedding")
            
        except Exception as e:
            logger.error(f"Error saving Q&A to Redis: {str(e)}")
            raise
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar Q&A pairs using vector similarity"""
        try:
            # Generate embedding for the query
            query_embedding = self._generate_embedding(query)
            
            # Prepare the Redis query
            embedding_bytes = self._vector_to_redis_format(query_embedding)
            
            # Perform vector search
            query_str = f'*=>[KNN {top_k} @embedding $embedding AS similarity]'
            params = {"embedding": embedding_bytes}
            
            # Execute the query
            results = self.redis.execute_command(
                'FT.SEARCH', 'qa_vector_idx', query_str, 
                'PARAMS', '2', 'embedding', embedding_bytes,
                'RETURN', '4', 'question', 'answer', 'category', 'similarity',
                'SORTBY', 'similarity', 'DESC'
            )
            
            if results and isinstance(results, list) and len(results) > 1:
                # Parse the results (Skip the first element which is the count)
                parsed_results = []
                for i in range(1, len(results), 2):  # Iterate through document IDs and properties
                    if i + 1 < len(results):
                        doc_fields = results[i + 1]
                        # Convert list of field-value pairs to dictionary
                        doc = {}
                        for j in range(0, len(doc_fields), 2):
                            if j + 1 < len(doc_fields):
                                field_name = doc_fields[j].decode('utf-8')
                                field_value = doc_fields[j + 1]
                                if isinstance(field_value, bytes):
                                    if field_name != 'embedding':  # Don't decode embedding bytes
                                        field_value = field_value.decode('utf-8')
                                doc[field_name] = field_value
                        
                        # Add formatted result with score
                        parsed_results.append({
                            "question": doc.get('question', ''),
                            "answer": doc.get('answer', ''),
                            "category": doc.get('category', ''),
                            "similarity": float(doc.get('similarity', 0.0))
                        })
                
                return parsed_results
            
            return []
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
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
                        "question": qa_pair.get(b"question", b"").decode(),
                        "answer": qa_pair.get(b"answer", b"").decode(),
                        "category": qa_pair.get(b"category", b"").decode(),
                        "created_at": qa_pair.get(b"created_at", b"").decode()
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
            vector_keys = self.redis.keys(f"{self.vector_key_prefix}*")
            
            # Delete all keys
            if question_keys:
                self.redis.delete(*question_keys)
            if category_keys:
                self.redis.delete(*category_keys)
            if vector_keys:
                self.redis.delete(*vector_keys)
            
            # Delete index
            self.redis.delete(self.index_key)
            
            # Drop vector index
            try:
                self.redis.execute_command("FT.DROPINDEX", "qa_vector_idx")
                # Recreate the vector index
                self._create_vector_index()
            except:
                pass
            
            logger.info("Successfully cleared all Q&A data")
            
        except Exception as e:
            logger.error(f"Error clearing Q&A data: {str(e)}")
            raise
    
    @staticmethod
    def _hash_question(question: str) -> str:
        """Generate a hash for the question"""
        import hashlib
        return hashlib.md5(question.encode()).hexdigest() 