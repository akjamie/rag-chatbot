import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

from config.common_settings import CommonConfig
from handler.tools.base_tool import BaseTool, ToolDescription, ToolCategory, ToolArgument
from utils.logging_util import logger
from utils.prompt_loader import load_txt_prompt

# Get absolute path to the project root
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

@dataclass
class QuestionContext:
    """Context for generating suggested questions"""
    original_query: str
    answer_content: str
    sources: List[Dict[str, Any]]
    chat_history: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict[str, Any]] = None

class SuggestedQuestionsGenerator(BaseTool):
    """Generate relevant follow-up questions based on context"""
    
    def __init__(self, llm: BaseChatModel, config: CommonConfig):
        super().__init__(config)
        self.llm = llm
        self.max_questions = 3
        self.logger = logger
        
        # Load prompt template
        prompt_path = os.path.join(PROJECT_ROOT, "handler", "prompts", "suggested_questions.txt")
        self.prompt_template = load_txt_prompt(
            prompt_path,
            input_variables=["max_questions", "original_query", "answer_content", "sources", "chat_history"]
        )

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="suggested_questions_generator",
            description="Generate relevant follow-up questions based on context",
            category=ToolCategory.OTHERS,
            arguments=[
                ToolArgument(
                    name="context",
                    description="QuestionContext containing query, answer, and history",
                    type="QuestionContext"
                )
            ]
        )

    def _validate_input(self, context: QuestionContext) -> bool:
        """Validate input context"""
        if not isinstance(context, QuestionContext):
            self.logger.error("Invalid input: context must be QuestionContext")
            return False
        if not context.original_query or not context.answer_content:
            self.logger.warning("Invalid context: missing query or answer content")
            return False
        return True

    def run(self, context: QuestionContext) -> List[str]:
        """Generate suggested follow-up questions"""
        if not self.enabled:
            return []
            
        try:
            if not self._validate_input(context):
                return []

            # Format prompt variables
            prompt_vars = {
                "max_questions": self.max_questions,
                "original_query": context.original_query,
                "answer_content": context.answer_content,
                "sources": self._format_sources(context.sources),
                "chat_history": self._format_chat_history(context.chat_history) if context.chat_history else ""
            }
            
            # Generate prompt using template
            prompt = self.prompt_template.format(**prompt_vars)
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            
            questions = self._process_questions(
                raw_questions=[q.strip() for q in response.split('\n') if q.strip()],
                context=context
            )
            
            self.logger.debug(f"Generated {len(questions)} suggested questions")
            return questions[:self.max_questions]
            
        except Exception as e:
            self.logger.error(f"Error generating suggested questions: {str(e)}")
            return []

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format source information for prompt context"""
        if not sources:
            return "No additional sources available."
            
        formatted_sources = []
        for source in sources:
            if "content" in source:
                formatted_sources.append(f"- {source['content'][:200]}...")
            elif "text" in source:
                formatted_sources.append(f"- {source['text'][:200]}...")
                
        return "\n".join(formatted_sources)
        
    def _format_chat_history(self, history: List[Dict[str, str]]) -> str:
        """Format chat history for context awareness"""
        if not history:
            return ""
            
        return "RECENT CHAT HISTORY:\n" + "\n".join(
            f"User: {exchange.get('user_input', '')}\n"
            f"Assistant: {exchange.get('response', '')}\n"
            for exchange in history[-2:]  # Only last 2 exchanges for relevance
        )
        
    def _process_questions(self, raw_questions: List[str], context: QuestionContext) -> List[str]:
        """Process and filter generated questions for quality and relevance"""
        processed_questions = []
        seen_concepts = set()
        
        for question in raw_questions:
            # Clean up the question
            question = self._clean_question(question)
            
            # Extract main concepts to check for duplicates
            main_concepts = self._extract_main_concepts(question)
            
            # Check if question is novel enough
            if not any(concept in seen_concepts for concept in main_concepts):
                seen_concepts.update(main_concepts)
                processed_questions.append(question)
                
        return processed_questions

    def _clean_question(self, question: str) -> str:
        """Clean and format a question"""
        # Remove question numbering
        question = re.sub(r'^\d+\.\s*', '', question)
        
        # Remove redundant question starters
        question = re.sub(r'^(Please tell me|Can you|I want to know|Tell me|I would like to know)\s+', '', question)
        
        # Convert indirect questions to direct ones
        question = re.sub(r'^I wonder\s+', '', question)
        
        return question.strip()
        
    def _extract_main_concepts(self, question: str) -> set:
        """Extract main concepts from question for duplicate detection"""
        words = question.lower().split()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 
                     'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for'}
        return set(word for word in words if word not in stop_words) 