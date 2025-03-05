from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

from config.common_settings import CommonConfig
from utils.logging_util import logger


class ToolCategory(Enum):
    """Core categories for RAG operations"""
    RETRIEVAL = "retrieval"  # Document retrieval
    GENERATION = "generation"  # Text generation
    EVALUATION = "evaluation"  # Response evaluation
    SEARCH = "web search"  # Web search
    FORMATTING = "formatting"  # Response formatting
    OTHERS = "others"

@dataclass
class ToolArgument:
    """Simplified but clear argument specification"""
    description: str
    type: str  # "str", "int", "dict", "list"
    example: Any
    optional: bool = False


@dataclass
class ToolDescription:
    """Practical tool description balancing completeness with ease of implementation"""
    name: str
    description: str
    category: ToolCategory
    args: Dict[str, ToolArgument]
    return_type: str

class BaseTool(ABC):
    def __init__(self, config: CommonConfig):
        self.config = config
        self.logger = logger
        self._validate_config()

    @property
    @abstractmethod
    def description(self) -> ToolDescription:
        """Return tool description metadata"""
        pass

    def get_tool_description(self) -> str:
        """Format tool description for LLM consumption"""
        desc = self.description
        formatted = f"""Tool: {desc.name} ({desc.category.value})
        Description: {desc.description}
        
        Arguments:
        {self._format_arguments(desc.args)}
        
        Returns type: {desc.return_type}
        """
        return formatted

    def _format_arguments(self, args: Dict[str, ToolArgument]) -> str:
        """Format arguments clearly but concisely"""
        formatted = []
        for name, arg in args.items():
            req_status = "optional" if arg.optional else "required"
            formatted.append(
                f"- {name} ({arg.type}, {req_status}): {arg.description}\n"
                f"  Example: {arg.example}"
            )
        return "\n".join(formatted)

    def validate_inputs(self, **kwargs) -> None:
        """Basic but effective input validation"""
        desc = self.description

        # Check required arguments
        required_args = {
            name: arg for name, arg in desc.args.items()
            if not arg.optional
        }

        missing_args = [
            name for name in required_args
            if name not in kwargs
        ]

        if missing_args:
            raise ValueError(
                f"Missing required arguments for {desc.name}: {', '.join(missing_args)}"
            )

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool's main functionality"""
        pass

    def is_enabled(self) -> bool:
        """Check if tool is enabled in configuration"""
        try:
            tool_name = self.description.name.lower().replace(" ", "_")
            return self.config.get_query_config(f"search.tools.{tool_name}.enabled", True)
        except Exception as e:
            self.logger.warning(f"Error checking if {self.description.name} is enabled: {str(e)}")
            return False

    def _validate_config(self):
        # Implementation of _validate_config method
        pass
