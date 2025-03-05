import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
import time

from config.common_settings import CommonConfig
from handler.tools.base_tool import BaseTool, ToolDescription, ToolCategory, ToolArgument
from utils.logging_util import logger


@dataclass
class CritiqueResult:
    """Result of response evaluation"""
    has_feedback: bool
    feedback: List[str]  # List of improvement suggestions
    enhanced_response: Optional[str] = None
    metadata: Dict[str, Any] = None


class ResponseCritic(BaseTool):
    """Evaluates response quality and enhances if needed"""

    def __init__(self, config: CommonConfig):
        super().__init__(config)
        self.llm = config.get_model("chatllm")
        self.logger = logger

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="response_critic",
            description="Critiques and enhances responses",
            category=ToolCategory.EVALUATION,
            args={
                "user_input": ToolArgument(type="str", description="Original user query"),
                "response": ToolArgument(type="str", description="Generated response to evaluate"),
                "context": ToolArgument(type="dict", description="Retrieved documents and chat history", example={
                    "documents": [
                        {"page_content": "...", "metadata": {"source": "...", "page": 1}},
                        {"page_content": "...", "metadata": {"source": "...", "page": 2}}
                    ],
                    "chat_history": [
                        {"user_input": "Where is the capital of France", "response": "The capital of France is Paris."},
                    ]
                })
            },
            return_type="str"
        )

    def run(self, **kwargs) -> str:
        """Main execution flow"""
        self.validate_inputs(**kwargs)

        try:
            # 1. Critique the response
            feedback = self._critique_response(
                user_input=kwargs["user_input"],
                response=kwargs["response"],
                context=kwargs["context"]
            )

            # 2. If there's feedback, enhance the response
            enhanced_response = None
            if feedback:
                enhanced_response = self._enhance_response(
                    user_input=kwargs["user_input"],
                    original_response=kwargs["response"],
                    context=kwargs["context"],
                    feedback=feedback
                )

            return enhanced_response

        except Exception as e:
            self.logger.error(f"Response critic failed: {str(e)}")
            return kwargs["response"]

    def _critique_response(self, user_input: str, response: str, context: Dict[str, Any]) -> List[str]:
        """Evaluate response quality using structured criteria"""

        prompt = f"""You are an expert response evaluator with deep experience in information accuracy and clarity.

        Role: Response Quality Assessor
        Task: Evaluate response quality and identify necessary improvements
        
        Input:
        User Query: {user_input}
        
        Generated Response: {response}
        
        Available Context:
        {self._format_context(context)}
        
        Evaluation Criteria:
        1. FACTUAL ACCURACY
        - Every claim must be supported by the context
        - No contradictions with provided information
        - No hallucinations or unsupported statements
        
        2. QUERY ALIGNMENT
        - Direct answer to the user's question
        - Addresses all aspects of the query
        - Maintains focus on user's intent
        
        3. COMPLETENESS
        - Uses relevant information from context
        - No critical information gaps
        - Balanced coverage of important points
        
        4. CLARITY & STRUCTURE
        - Clear and logical flow
        - Appropriate level of detail
        - Professional and coherent presentation
        
        Instructions:
        - If ANY criteria are not met, provide specific feedback points
        - Each feedback point must be actionable and specific
        - If response meets ALL criteria, return an empty list
        - Focus on substance over style
        
        Output Format:
        Return ONLY a list of feedback points, one per line.
        If no improvements needed, return nothing.
        
        Example Feedback:
        - Missing key information about X from paragraph 2
        - Claim about Y contradicts source document
        - Question about Z remains unaddressed
        - Structure needs better organization for clarity"""

        try:
            result = self.llm.invoke([HumanMessage(content=prompt)]).content
            return [line.strip() for line in result.split('\n') if line.strip() and line.startswith('-')]
        except Exception as e:
            self.logger.error(f"Critique failed: {str(e)}")
            return []

    def _enhance_response(self, user_input: str, original_response: str,
                          context: Dict[str, Any], feedback: List[str]) -> str:
        """Generate improved response addressing specific feedback"""

        prompt = f"""You are an expert content enhancer focused on accuracy and clarity.

        Role: Response Enhancement Specialist
        Task: Improve the response while maintaining accuracy and addressing feedback
        
        Input Context:
        Original Query: {user_input}
        
        Current Response: {original_response}
        
        Available Information:
        {self._format_context(context)}
        
        Improvement Requirements:
        {chr(10).join(f"- {point}" for point in feedback)}
        
        Enhancement Guidelines:
        1. ACCURACY FIRST
        - Every statement must be supported by context
        - Maintain factual precision
        - No speculation or unsupported claims
        
        2. COMPREHENSIVE IMPROVEMENT
        - Address ALL feedback points
        - Maintain existing correct information
        - Fill identified information gaps
        
        3. CLARITY ENHANCEMENT
        - Improve structure where needed
        - Maintain professional tone
        - Ensure logical flow
        
        4. CONTEXT INTEGRATION
        - Incorporate relevant context naturally
        - Maintain appropriate detail level
        - Ensure balanced coverage
        
        Instructions:
        - Generate a complete, enhanced response
        - Address all feedback points systematically
        - Maintain or improve existing strengths
        - Focus on substantial improvements
        
        Generate the enhanced response now, try to ensure the output length does not exceed 200 characters"""

        try:
            return self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            self.logger.error(f"Enhancement failed: {str(e)}, stack:{traceback.format_exc()}")
            return original_response

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context with clear structure and priority"""
        parts = []

        # Add retrieved documents with relevance scores
        if docs := context.get("documents", []):
            parts.append("Retrieved Information (by relevance):")
            for idx, doc in enumerate(docs[:3], 1):  # Top 3 most relevant
                score = doc.metadata.get('relevance_score', 0)
                source = doc.metadata.get('source', 'Unknown')
                parts.append(f"{idx}. [{score:.2f}] From {source}:")
                parts.append(f"   {doc.content[:200]}...")

        # Add recent chat history with timestamps
        if history := context.get("chat_histories", []):
            parts.append("\nRelevant Conversation Context:")
            for turn in history[-2:]:  # Last 2 turns
                timestamp = turn.created_at.strftime("%Y-%m-%d %H:%M:%S")
                parts.append(f"[{timestamp}]")
                parts.append(f"User: {turn.user_input}")
                parts.append(f"Assistant: {turn.response}")

        return "\n".join(parts)
