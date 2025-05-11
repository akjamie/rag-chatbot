import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from config.common_settings import CommonConfig
from utils.logging_util import logger


class ParentChildDocumentRetriever:
    """
    A retriever that handles parent-child document relationships.
    It focuses on retrieving both parent and relevant child documents
    based on a query without the document addition functionality.
    """
    
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore, config: CommonConfig):
        """
        Initialize the ParentChildDocumentRetriever.
        
        Args:
            llm: The language model to use for reranking (if enabled)
            vectorstore: The vector store containing indexed documents
            config: Configuration object with retriever settings
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.config = config
        self.logger = logger
        
        # Configure parent-child settings from config
        search_config = self.config.get_query_config("search")
        self.parent_child_enabled = search_config.get("parent_child_enabled", False)
        
        # Only load these configurations if parent-child mode is enabled
        if self.parent_child_enabled:
            self.parent_chunk_size = search_config.get("parent_chunk_size", 2000)
            self.child_chunk_size = search_config.get("child_chunk_size", 400)
            self.chunk_overlap = search_config.get("chunk_overlap", 50)
            self.batch_size = search_config.get("batch_size", 32)
        
        # Configure reranking if enabled
        self.rerank_enabled = config.get_query_config("search.rerank_enabled", False)
        self.reranker = config.get_model("rerank") if self.rerank_enabled else None
        
    def _get_parent_ids_from_children(self, child_documents: List[Document]) -> List[str]:
        """
        Extracts parent document IDs from child documents' metadata.
        
        Args:
            child_documents: List of child documents retrieved from a search
            
        Returns:
            List of unique parent document IDs
        """
        parent_ids = []
        
        for doc in child_documents:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)
        
        return parent_ids
    
    def _retrieve_parent_documents(self, parent_ids: List[str]) -> List[Document]:
        """
        Retrieves parent documents based on their IDs.
        
        Args:
            parent_ids: List of parent document IDs to retrieve
            
        Returns:
            List of parent documents
        """
        parent_documents = []
        
        try:
            # This would typically involve a direct retrieval from storage
            # Since we're not implementing document addition, we'll retrieve from vectorstore
            for parent_id in parent_ids:
                # Use metadata filtering if the vectorstore supports it
                if hasattr(self.vectorstore, "get_relevant_documents"):
                    filters = {"doc_id": parent_id, "is_parent": True}
                    docs = self.vectorstore.get_relevant_documents(
                        "", metadata=filters
                    )
                    parent_documents.extend(docs)
                    
        except Exception as e:
            self.logger.error(f"Error retrieving parent documents: {str(e)}")
            self.logger.debug(traceback.format_exc())
        
        return parent_documents
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Reranks documents based on relevance to the query using the reranker model.
        
        Args:
            query: The user query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not self.rerank_enabled or not self.reranker or not documents:
            return documents
            
        try:
            pairs = []
            for doc in documents:
                pairs.append([query, doc.page_content])
                
            scores = self.reranker.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Combine documents with scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Add scores to metadata
            for doc, score in scored_docs:
                doc.metadata["rerank_score"] = score
                
            return [doc for doc, _ in scored_docs]
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return documents
    
    def run(self, query: str, relevance_threshold: float = 0.7, max_documents: int = 5) -> List[Document]:
        """
        Run retrieval for parent-child documents based on the query.
        
        Args:
            query: The user query
            relevance_threshold: Minimum relevance score threshold
            max_documents: Maximum number of documents to retrieve
            
        Returns:
            List of parent documents, sorted by relevance
        """
        self.logger.info(f"Running parent-child document retrieval with query: {query}")
        
        if not self.parent_child_enabled:
            self.logger.warning("Parent-child retrieval is disabled in configuration")
            return []
            
        try:
            # First, retrieve the most relevant child documents
            child_documents = self.vectorstore.similarity_search_with_score(
                query=query,
                k=max_documents * 2  # Retrieve more children to find good parents
            )
            
            # Extract and parse out the scores
            scored_children = []
            for doc, score in child_documents:
                doc.metadata["vector_score"] = score
                scored_children.append(doc)
                
            # Get parent IDs from the child documents
            parent_ids = self._get_parent_ids_from_children(scored_children)
            
            if not parent_ids:
                self.logger.warning("No parent documents found from child documents")
                return []
                
            # Retrieve the parent documents
            parent_documents = self._retrieve_parent_documents(parent_ids)
            
            if not parent_documents:
                self.logger.warning("Failed to retrieve parent documents")
                return []
                
            # Rerank the parent documents if reranking is enabled
            if self.rerank_enabled:
                parent_documents = self._rerank_documents(query, parent_documents)
                
            # Filter by threshold if specified
            if relevance_threshold > 0:
                # For reranked documents
                if self.rerank_enabled:
                    parent_documents = [
                        doc for doc in parent_documents 
                        if doc.metadata.get("rerank_score", 0) >= relevance_threshold
                    ]
                # For vector similarity scores
                else:
                    parent_documents = [
                        doc for doc in parent_documents 
                        if doc.metadata.get("vector_score", 0) >= relevance_threshold
                    ]
            
            # Limit to max_documents
            return parent_documents[:max_documents]
            
        except Exception as e:
            self.logger.error(f"Error in parent-child document retrieval: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise


if __name__ == "__main__":
    """
    用于测试ParentChildDocumentRetriever功能的简单示例代码
    """
    try:
        # 初始化配置
        config = CommonConfig()
        # 获取向量存储
        vectorstore = config.get_vector_store()
        # 获取语言模型
        llm = config.get_model("chatllm")
        
        # 初始化检索器
        retriever = ParentChildDocumentRetriever(llm, vectorstore, config)
        
        # 测试查询
        query = "如何使用RAG技术提高问答系统的准确性?"
        print(f"测试查询: {query}")
        
        # 运行检索
        documents = retriever.run(query)
        
        # 打印结果
        print(f"共检索到 {len(documents)} 个父文档")
        for i, doc in enumerate(documents, 1):
            print(f"\n文档 {i}:")
            print(f"内容: {doc.page_content[:200]}...")
            
            # 打印分数
            if retriever.rerank_enabled:
                score = doc.metadata.get("rerank_score", "未知")
                print(f"重排序分数: {score}")
            else:
                score = doc.metadata.get("vector_score", "未知")
                print(f"向量相似度分数: {score}")
                
            # 打印元数据
            print(f"元数据: {doc.metadata}")
            
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        print(traceback.format_exc()) 