from typing import List, Dict, Any
from dataclasses import dataclass

from config.common_settings import CommonConfig
from handler.tools.base_tool import BaseTool, ToolDescription, ToolCategory, ToolArgument
from utils.logging_util import logger

@dataclass
class CitationContext:
    """Context for citation extraction"""
    documents: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]] = None

class CitationExtractor(BaseTool):
    """Extract citations from documents and web search results"""
    
    def __init__(self, config: CommonConfig):
        super().__init__(config)
        self.enabled = config.get_query_config("features.citations_enabled", True)
        self.logger = logger

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="citation_extractor",
            description="Extract source citations from documents and web results, can be the last step",
            category=ToolCategory.OTHERS,
            arguments=[
                ToolArgument(
                    name="context",
                    description="CitationContext containing documents and web results",
                    type="CitationContext"
                )
            ]
        )

    def _validate_input(self, context: CitationContext) -> bool:
        """Validate input context"""
        if not isinstance(context, CitationContext):
            self.logger.error("Invalid input: context must be CitationContext")
            return False
        if not context.documents and not context.web_results:
            self.logger.warning("Empty context: no documents or web results provided")
            return False
        return True

    def run(self, context: CitationContext) -> List[Dict[str, str]]:
        """Extract citations from provided context"""
        if not self.enabled:
            return []

        try:
            if not self._validate_input(context):
                return []

            citations = []
            
            # Extract document sources
            if context.documents:
                doc_sources = {
                    doc.metadata.get("source") 
                    for doc in context.documents 
                    if doc.metadata.get("source")
                }
                citations.extend([{"source": source} for source in doc_sources])

            # Extract web sources
            if context.web_results:
                web_sources = {
                    result.get("url") 
                    for result in context.web_results 
                    if result.get("url")
                }
                citations.extend([{"source": url} for url in web_sources])

            # Remove duplicates while preserving order
            unique_citations = []
            seen_sources = set()
            for citation in citations:
                source = citation["source"]
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_citations.append(citation)

            self.logger.debug(f"Extracted {len(unique_citations)} unique citations")
            return unique_citations

        except Exception as e:
            self.logger.error(f"Error extracting citations: {str(e)}")
            return []

    def _is_valid_source(self, source: str) -> bool:
        """Validate source string"""
        if not source:
            return False
        if len(source) > 500:  # Reasonable max length for a source
            return False
        return True 