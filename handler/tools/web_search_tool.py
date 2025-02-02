from typing import List, Dict, Any
from langchain_community.tools import TavilySearchResults

from config.common_settings import CommonConfig
from utils.logging_util import logger


class WebSearch:
    def __init__(self, config: CommonConfig):
        self.logger = logger
        self.config = config

        if config.get_query_config("search.web_search_enabled", False):
            self.logger.info("Web search is enabled")
            self.web_search_tool = TavilySearchResults(
                k=config.get_query_config("limits.max_web_results", 3)
            )
        else:
            self.logger.info("Web search is disabled")

    def run(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute web search with error handling and logging
        
        Args:
            query: Search query string
            
        Returns:
            List of search results, each containing:
                - title: Title of the webpage
                - url: URL of the source
                - content: Snippet of relevant content
        """
        try:
            # search use original query, without much chat histories added
            self.logger.info(f"Running web search for query: {query}")

            if not self.config.get_query_config("search.web_search_enabled", False):
                self.logger.info("Web search is disabled")
                return []
            results = self.web_search_tool.invoke(query)
            
            if not results:
                self.logger.warning(f"No results found for query: {query}")
                return []
            

            self.logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during web search: {str(e)}")
            return []
