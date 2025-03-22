from typing import Optional, Dict, Any, List
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logging_util import logger
from config.common_settings import CommonConfig

class RedisQueryTool:
    """Tool for querying Q&A pairs from Redis using vector search"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize Redis query tool with vector search capabilities"""
        if redis_client is None:
            config = CommonConfig()
            redis_client = redis.Redis(
                host=config.get_env_variable("REDIS_HOST", "localhost"),
                port=int(config.get_env_variable("REDIS_PORT", 6379)),
                db=int(config.get_env_variable("REDIS_DB", 0)),
                decode_responses=False  # Need raw bytes for vector operations
            )
        self.redis = redis_client
        self.question_key_prefix = "qa:question:"
        self.vector_key_prefix = "qa:vector:"
        self.index_key = "qa:index:questions"
        self.similarity_threshold = 0.75  # 75% similarity threshold
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text using sentence transformer"""
        if not text or not isinstance(text, str):
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding
    
    def _vector_to_redis_format(self, vector: np.ndarray) -> bytes:
        """Convert numpy vector to Redis byte format"""
        return vector.astype(np.float32).tobytes()
    
    def _vector_search(self, query_embedding: np.ndarray, top_k: int = 1) -> Optional[Dict[str, Any]]:
        """Search for similar questions using vector similarity"""
        try:
            # Convert embedding to bytes for Redis
            embedding_bytes = self._vector_to_redis_format(query_embedding)
            
            # Perform vector search using Redis search
            query_str = f'*=>[KNN {top_k} @embedding $embedding AS similarity]'
            
            # Execute the search command
            results = self.redis.execute_command(
                'FT.SEARCH', 'qa_vector_idx', query_str, 
                'PARAMS', '2', 'embedding', embedding_bytes,
                'RETURN', '4', 'question', 'answer', 'category', 'similarity',
                'SORTBY', 'similarity', 'DESC',
                'LIMIT', '0', str(top_k)
            )
            
            # Process the results
            if results and isinstance(results, list) and len(results) > 1:
                # Skip the first element which is the count
                if results[0] > 0:
                    # Get the first result (highest similarity)
                    doc_fields = results[2]  # Fields of the first result
                    
                    # Extract field values
                    doc = {}
                    for i in range(0, len(doc_fields), 2):
                        if i + 1 < len(doc_fields):
                            field_name = doc_fields[i].decode('utf-8')
                            field_value = doc_fields[i + 1]
                            if isinstance(field_value, bytes) and field_name != 'embedding':
                                field_value = field_value.decode('utf-8')
                            doc[field_name] = field_value
                    
                    # Get similarity score
                    similarity = float(doc.get('similarity', 0.0))
                    
                    # Check if similarity meets threshold
                    if similarity >= self.similarity_threshold:
                        return {
                            "question": doc.get('question', ''),
                            "answer": doc.get('answer', ''),
                            "category": doc.get('category', ''),
                            "similarity": similarity
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return None
    
    def query(self, question: str) -> Optional[Dict[str, Any]]:
        """Query Redis for an answer to the given question using vector search"""
        if not question or not isinstance(question, str) or len(question) > 500:
            logger.warning("Invalid question format or length")
            return None
            
        try:
            # First try exact match by hash
            question_hash = self._hash_question(question)
            key = f"{self.question_key_prefix}{question_hash}"
            
            if self.redis.exists(key):
                # Get Q&A pair from Redis
                qa_data = self.redis.hgetall(key)
                return {
                    "question": qa_data.get(b"question", b"").decode('utf-8'),
                    "answer": qa_data.get(b"answer", b"").decode('utf-8'),
                    "category": qa_data.get(b"category", b"").decode('utf-8'),
                    "similarity": 1.0
                }
            
            # If no exact match, try vector search
            query_embedding = self._generate_embedding(question)
            return self._vector_search(query_embedding)
            
        except Exception as e:
            logger.error(f"Error querying Redis: {str(e)}")
            return None
    
    def batch_query(self, questions: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Batch query multiple questions and return top k results for each"""
        results = []
        
        for question in questions:
            try:
                # Generate embedding
                query_embedding = self._generate_embedding(question)
                
                # Get multiple results
                embedding_bytes = self._vector_to_redis_format(query_embedding)
                
                # Execute vector search
                search_results = self.redis.execute_command(
                    'FT.SEARCH', 'qa_vector_idx', f'*=>[KNN {top_k} @embedding $embedding AS similarity]', 
                    'PARAMS', '2', 'embedding', embedding_bytes,
                    'RETURN', '4', 'question', 'answer', 'category', 'similarity',
                    'SORTBY', 'similarity', 'DESC',
                    'LIMIT', '0', str(top_k)
                )
                
                # Process results
                if search_results and search_results[0] > 0:
                    question_results = []
                    for i in range(1, len(search_results), 2):
                        if i + 1 < len(search_results):
                            doc_fields = search_results[i + 1]
                            doc = {}
                            for j in range(0, len(doc_fields), 2):
                                if j + 1 < len(doc_fields):
                                    field_name = doc_fields[j].decode('utf-8')
                                    field_value = doc_fields[j + 1]
                                    if isinstance(field_value, bytes) and field_name != 'embedding':
                                        field_value = field_value.decode('utf-8')
                                    doc[field_name] = field_value
                            
                            similarity = float(doc.get('similarity', 0.0))
                            question_results.append({
                                "question": doc.get('question', ''),
                                "answer": doc.get('answer', ''),
                                "category": doc.get('category', ''),
                                "similarity": similarity
                            })
                    
                    results.append({
                        "query": question,
                        "results": question_results
                    })
                else:
                    results.append({
                        "query": question,
                        "results": []
                    })
                    
            except Exception as e:
                logger.error(f"Error in batch query for question '{question}': {str(e)}")
                results.append({
                    "query": question,
                    "results": [],
                    "error": str(e)
                })
        
        return results
    
    @staticmethod
    def _hash_question(question: str) -> str:
        """Generate a hash for the question"""
        import hashlib
        return hashlib.md5(question.lower().encode()).hexdigest() 