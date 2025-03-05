import traceback
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from config.common_settings import CommonConfig


class QueryExpander:
    def __init__(self, llm: BaseChatModel, config: CommonConfig):
        self.config = config
        self.llm = llm

    def run(self, query: str) -> List[str]:
        try:
            self.logger.info(f"Expanding query: {query}")
            
            # Using the original tested prompt
            prompt = f"""Generate 3 alternative search queries that are semantically similar to the original query.
            Each query should:
            - Maintain the same core intent and topic
            - Use different but related terms/synonyms
            - Be specific and searchable
            - Be a complete, well-formed question
            - Follow the same question type (if/what/how/why/when/where) as the original
            - Be approximately the same length as the original
            - Not introduce new concepts not present in the original
            
            Format rules:
            - Each query must be a single line
            - No prefixes, numbers, or bullet points
            - No explanations or additional text or numbering
            - No quotes or special characters
            - End each query with a question mark if it's a question
            
            Original query: "{query}"
            
            Return exactly 3 queries, one per line:"""

            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            expanded_queries = [q.strip() for q in response.split('\n') if q.strip()]
            return expanded_queries[:3]  #

        except Exception as e:
            self.logger.error(f"Error in query expansion: {str(e)}, stack: {traceback.format_exc()}")
            return []

