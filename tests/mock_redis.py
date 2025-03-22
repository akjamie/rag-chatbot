from typing import Optional, Dict, Any, Set, List, Union, Tuple
from collections import defaultdict
from utils.logging_util import logger
import hashlib
from datetime import datetime
import logging
import json
import os
import numpy as np
from numpy.linalg import norm

class MockRedis:
    """Mock Redis implementation for local debugging"""
    
    def __init__(self):
        self.data = {}  # Key-value store
        self.sets = defaultdict(set)  # Set store
        self.hashes = defaultdict(dict)  # Hash store
        self.vector_store = {}  # Store for vector embeddings
    
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
    
    def hgetall(self, key: str) -> Dict[str, str]:
        result = {}
        for k, v in self.hashes.get(key, {}).items():
            if isinstance(k, str):
                result[k.encode()] = v.encode() if isinstance(v, str) else v
            else:
                result[k] = v
        return result
    
    def hmset(self, key: str, mapping: Dict[str, Any]) -> bool:
        if key not in self.hashes:
            self.hashes[key] = {}
        for field, value in mapping.items():
            self.hashes[key][field] = value
        return True
    
    def execute_command(self, *args):
        """Mock Redis command execution for vector operations"""
        cmd = args[0].upper() if args else ""
        
        if cmd == "FT._LIST":
            return [b"qa_vector_idx"]
        
        elif cmd == "FT.CREATE":
            # Just return OK for index creation
            return "OK"
        
        elif cmd == "FT.SEARCH":
            # Handle vector search
            if len(args) < 3 or not args[2].endswith('AS similarity]'):
                return [0]  # No results
            
            # Parse command args to get the embedding bytes
            embedding_bytes = None
            for i, arg in enumerate(args):
                if arg == 'embedding' and i+1 < len(args):
                    embedding_bytes = args[i+1]
                    break
            
            if not embedding_bytes:
                return [0]  # No embedding provided
                
            # Convert bytes to numpy array
            query_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Find top matches in vector store
            matches = []
            for key, vector_data in self.vector_store.items():
                if 'embedding' in vector_data:
                    stored_embedding = vector_data['embedding']
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    matches.append((key, similarity, vector_data))
            
            # Sort by similarity (descending)
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Extract limit params
            limit_start = 0
            limit_count = 1
            for i, arg in enumerate(args):
                if arg == 'LIMIT' and i+2 < len(args):
                    limit_start = int(args[i+1])
                    limit_count = int(args[i+2])
                    break
            
            # Apply limit
            matches = matches[limit_start:limit_start+limit_count]
            
            if not matches:
                return [0]  # No matches found
            
            # Format result as Redis would
            result = [len(matches)]  # Count of matches
            
            for i, (key, similarity, data) in enumerate(matches):
                # Add document ID
                result.append(key)
                
                # Add document fields
                fields = []
                for field_name, field_value in data.items():
                    if field_name == 'embedding':
                        continue  # Skip embedding in results
                    
                    # Convert field name and value to bytes if they aren't already
                    fields.append(field_name.encode() if isinstance(field_name, str) else field_name)
                    if isinstance(field_value, str):
                        fields.append(field_value.encode())
                    elif isinstance(field_value, (int, float)):
                        fields.append(str(field_value).encode())
                    else:
                        fields.append(field_value)
                
                # Add similarity score
                fields.append(b'similarity')
                fields.append(str(similarity).encode())
                
                result.append(fields)
            
            return result
        
        # Default fallback
        return None
    
    def ping(self) -> bool:
        return True
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

class MockVectorSearchTool:
    """Mock vector search tool for testing semantic search"""
    
    def __init__(self):
        self.redis = MockRedis()
        self.question_key_prefix = "qa:question:"
        self.vector_key_prefix = "qa:vector:"
        self.index_key = "qa:index:questions"
        self.similarity_threshold = 0.7
        self.embedding_dim = 384  # Common dimension for embeddings
        self._load_qa_pairs()
    
    def _hash_question(self, question: str) -> str:
        """Generate a hash for the question"""
        return hashlib.md5(question.encode()).hexdigest()
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic embedding based on text content"""
        if not text or not isinstance(text, str):
            return np.zeros(self.embedding_dim)
        
        # Normalize text
        text = text.lower()
        for char in '.,!?;:()[]{}"\'':
            text = text.replace(char, ' ')
        
        # Create a deterministic embedding
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
        np.random.seed(seed)
        embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        
        # Modify embedding based on key terms
        word_vectors = {
            'identity': np.array([0.8, 0.2, 0.1] + [0] * (self.embedding_dim - 3)),
            'management': np.array([0.2, 0.7, 0.3] + [0] * (self.embedding_dim - 3)),
            'system': np.array([0.1, 0.3, 0.7] + [0] * (self.embedding_dim - 3)),
            'information': np.array([0.7, 0.1, 0.2] + [0] * (self.embedding_dim - 3)),
            'technology': np.array([0.3, 0.2, 0.8] + [0] * (self.embedding_dim - 3)),
            'operation': np.array([0.2, 0.8, 0.2] + [0] * (self.embedding_dim - 3)),
            'platform': np.array([0.1, 0.1, 0.9] + [0] * (self.embedding_dim - 3)),
            'idam': np.array([0.9, 0.8, 0.7] + [0] * (self.embedding_dim - 3)),
            'itop': np.array([0.7, 0.9, 0.8] + [0] * (self.embedding_dim - 3)),
            'vector': np.array([0.6, 0.7, 0.9] + [0] * (self.embedding_dim - 3)),
            'search': np.array([0.8, 0.6, 0.7] + [0] * (self.embedding_dim - 3)),
            'database': np.array([0.7, 0.5, 0.8] + [0] * (self.embedding_dim - 3)),
            'semantic': np.array([0.9, 0.6, 0.5] + [0] * (self.embedding_dim - 3)),
        }
        
        for word in text.split():
            if word in word_vectors:
                embedding += word_vectors[word] * 0.2
        
        # Normalize embedding
        embedding_norm = norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        
        return embedding
    
    def query(self, question: str) -> Optional[Dict[str, Any]]:
        """Query the mock vector database for an answer"""
        if not question or not isinstance(question, str) or len(question) > 500:
            return None
            
        try:
            # Try exact match first
            question_hash = self._hash_question(question)
            key = f"{self.question_key_prefix}{question_hash}"
            
            if self.redis.exists(key):
                data = self.redis.hgetall(key)
                return {
                    "question": data.get(b"question", b"").decode() if isinstance(data.get(b"question", b""), bytes) else data.get("question", ""),
                    "answer": data.get(b"answer", b"").decode() if isinstance(data.get(b"answer", b""), bytes) else data.get("answer", ""),
                    "category": data.get(b"category", b"").decode() if isinstance(data.get(b"category", b""), bytes) else data.get("category", ""),
                    "similarity": 1.0
                }
            
            # Try vector search
            query_embedding = self._generate_embedding(question)
            
            # Convert embedding to bytes
            embedding_bytes = query_embedding.astype(np.float32).tobytes()
            
            # Use mock FT.SEARCH command
            results = self.redis.execute_command(
                'FT.SEARCH', 'qa_vector_idx', f'*=>[KNN 1 @embedding $embedding AS similarity]', 
                'PARAMS', '2', 'embedding', embedding_bytes,
                'RETURN', '4', 'question', 'answer', 'category', 'similarity',
                'SORTBY', 'similarity', 'DESC',
                'LIMIT', '0', '1'
            )
            
            if results and results[0] > 0:
                doc_fields = results[2]  # Fields of the first result
                
                # Extract fields
                doc = {}
                for i in range(0, len(doc_fields), 2):
                    if i + 1 < len(doc_fields):
                        field_name = doc_fields[i].decode() if isinstance(doc_fields[i], bytes) else doc_fields[i]
                        field_value = doc_fields[i + 1]
                        if isinstance(field_value, bytes):
                            field_value = field_value.decode()
                        doc[field_name] = field_value
                
                similarity = float(doc.get('similarity', 0.0))
                
                if similarity >= self.similarity_threshold:
                    return {
                        "question": doc.get('question', ''),
                        "answer": doc.get('answer', ''),
                        "category": doc.get('category', ''),
                        "similarity": similarity
                    }
            
            return None
            
        except Exception as e:
            logging.error(f"Error in query: {str(e)}")
            return None
    
    def add_qa_with_embedding(self, question: str, answer: str, category: str, 
                             embedding: Optional[np.ndarray] = None) -> None:
        """Add a Q&A pair with custom embedding to the mock vector database"""
        try:
            if embedding is not None and embedding.shape != (self.embedding_dim,):
                raise ValueError(f"Embedding must have shape ({self.embedding_dim},)")
                
            question_hash = self._hash_question(question)
            key = f"{self.question_key_prefix}{question_hash}"
            vector_key = f"{self.vector_key_prefix}{question_hash}"
            
            # Generate embedding if not provided
            if embedding is None:
                embedding = self._generate_embedding(question)
            
            # Store Q&A pair
            qa_data = {
                "question": question,
                "answer": answer,
                "category": category,
                "created_at": datetime.now().isoformat()
            }
            self.redis.hmset(key, qa_data)
            
            # Store vector data
            vector_data = {
                "question": question,
                "answer": answer,
                "category": category,
                "embedding": embedding
            }
            self.redis.hmset(vector_key, qa_data)
            self.redis.vector_store[vector_key] = vector_data
            
            # Add to index
            self.redis.sadd(self.index_key, question_hash)
            
        except Exception as e:
            logging.error(f"Error adding Q&A pair: {str(e)}")
            raise
    
    def _load_qa_pairs(self) -> None:
        """Load Q&A pairs for testing"""
        # Default Q&A pairs
        default_qa_pairs = [
            {
                "question": "IDAM",
                "answer": "IDAM is identity and access management system",
                "category": "geography"
            },
            {
                "question": "ITOP",
                "answer": "ITOP is information technology operation platform",
                "category": "technology"
            },
            {
                "question": "What is RAG?",
                "answer": "RAG (Retrieval-Augmented Generation) is a technique that enhances language models by retrieving external information before generating responses.",
                "category": "technology"
            },
            {
                "question": "Vector databases",
                "answer": "Vector databases are specialized databases designed to store and search vector embeddings efficiently using similarity metrics like cosine similarity.",
                "category": "technology"
            }
        ]
        
        # Add each default Q&A pair
        for qa_pair in default_qa_pairs:
            embedding = self._generate_embedding(qa_pair["question"])
            self.add_qa_with_embedding(
                question=qa_pair["question"],
                answer=qa_pair["answer"],
                category=qa_pair["category"],
                embedding=embedding
            ) 