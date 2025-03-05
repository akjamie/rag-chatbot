from typing import List, Dict, Any
from langchain_community.tools import TavilySearchResults

from config.common_settings import CommonConfig
from handler.tools.base_tool import BaseTool, ToolDescription, ToolArgument, ToolCategory
from utils.logging_util import logger


class TavilyWebSearch(BaseTool):
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

    def run(self, **kwargs) -> List[Dict[str, Any]]:
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
            self.validate_inputs(**kwargs)
            query = kwargs["query"]
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
    
    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="tavily_web_search",
            description="AI-powered web search using Tavily API",
            category=ToolCategory.SEARCH,
            args={
                "query": ToolArgument(
                    description="The search query to execute",
                    type="str",
                    example="What is the capital of France?",
                    optional=False
                )
            },
            return_type="List[Dict[str, Any]]"
        )

