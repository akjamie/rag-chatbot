from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from pydantic import create_model, BaseModel

from config.common_settings import CommonConfig
from conversation.conversation_history_helper import ConversationHistoryHelper
from conversation.repositories import ConversationHistoryRepository
from handler.tools.citation_extractor import CitationExtractor
from handler.tools.document_retriever import DocumentRetriever
from handler.tools.hypothetical_answer import HypotheticalAnswerGenerator
from handler.tools.response_critic import ResponseCritic
from handler.tools.response_formatter import ResponseFormatter
from handler.tools.suggested_questions_generator import SuggestedQuestionsGenerator
from handler.tools.tavily_web_search import TavilyWebSearch
from handler.utils.llm_message_tracker import LLMMessageTracker, LLMInteraction
from utils.logging_util import logger
from handler.tools.base_tool import BaseTool, ToolDescription, ToolArgument


@dataclass
class QueryResponse:
    answer: str
    citations: List[Dict[str, Any]]
    suggested_questions: List[str]
    metadata: Dict[str, Any]


@dataclass
class AgentState:
    """Track the state of agent's reasoning and execution"""
    current_step: str
    thought: str
    observation: Optional[str] = None
    tools_used: List[str] = None
    error: Optional[str] = None


class ReActAgent:
    """Advanced ReAct agent with enhanced reasoning and tool usage"""

    def __init__(self, llm: BaseChatModel, config: CommonConfig):
        self.llm = llm
        self.config = config
        self.logger = logger

        # Initialize base tools
        base_tools = [
            DocumentRetriever(self.config),
            TavilyWebSearch(self.config),
            HypotheticalAnswerGenerator(self.config),
            CitationExtractor(self.config),
            ResponseFormatter(self.config),
            ResponseCritic(self.config),
            SuggestedQuestionsGenerator(self.config)
        ]
        
        # Convert to LangChain tools
        self.tools = self._convert_to_langchain_tools(base_tools)
        
        # Initialize other components
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        self.agent_executor = self._create_agent_executor()

        # Initialize reasoning tracker
        self.reasoning_log = []

        self.conversation_helper = ConversationHistoryHelper(
            ConversationHistoryRepository(config.get_db_manager())
        )
        self.llm_tracker = LLMMessageTracker()

    def _convert_to_langchain_tools(self, base_tools: List[BaseTool]) -> List[Tool]:
        """Convert our BaseTool instances to LangChain Tool format"""
        langchain_tools = []
        self.tool_map = {}  # Keep reference to original tools
        
        for tool in base_tools:
            if not tool.is_enabled:
                continue
            
            desc = tool.description
            
            # Use the tool's run method directly
            tool_func = tool.run
            
            langchain_tool = Tool(
                name=desc.name,
                description=desc.description,
                func=tool_func,
                args_schema=self._create_args_schema(desc.arguments),
                return_direct=False
            )
            
            langchain_tools.append(langchain_tool)
            self.tool_map[desc.name] = tool
            
        return langchain_tools

    def _create_args_schema(self, arguments: List[ToolArgument]) -> Type[BaseModel]:
        """Create Pydantic model for tool arguments"""
        fields = {}
        for arg in arguments:
            python_type = self._get_python_type(arg.type)
            fields[arg.name] = (python_type, ...)  # ... means required field
            
        return create_model('ToolArgs', **fields)

    def _get_python_type(self, type_str: str) -> Type:
        """Convert string type to Python type"""
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'QuestionContext': dict,
            'CitationContext': dict
        }
        return type_mapping.get(type_str, str)

    def _create_agent_executor(self) -> AgentExecutor:
        """Create agent executor with optimized prompt engineering"""
        
        system_prompt = """You are an intelligent research assistant with access to the following tools:

        {tools}
        
        Follow these steps for every query:
        1. ANALYZE: Understand the query intent and required information
        2. PLAN: Determine which tools to use and in what order
        3. EXECUTE: Use tools systematically to gather information
        4. SYNTHESIZE: Combine information into a coherent answer
        5. VALIDATE: Check accuracy and completeness
        6. FORMAT: Structure the response with:
           - Clear answer
        
        Always explain your thinking process using "Reasoning:" before each major step."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=self.config.get_query_config("agent.verbose", True),
            handle_parsing_errors=True,
            max_iterations=self.config.get_query_config("agent.max_iterations", 5),
            early_stopping_method="force",
            return_intermediate_steps=True
        )

    async def process_query(self,
                            query: str,
                            user_id: str,
                            session_id: str,
                            request_id: str) -> QueryResponse:
        try:
            # Get conversation history
            history = self.conversation_helper.get_conversation_history(
                user_id=user_id,
                session_id=session_id,
                limit=5  # Last 5 interactions
            )

            # Convert history to memory format
            memory_messages = []
            for msg in history:
                memory_messages.extend([
                    HumanMessage(content=msg.user_input),
                    AIMessage(content=msg.response)
                ])

            # Update memory
            self.memory.chat_memory.messages.extend(memory_messages)

            # Process query with memory context
            result = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": memory_messages
            })

            # Track LLM interactions
            self.llm_tracker.track(LLMInteraction(
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                prompt=query,
                response=result["output"],
                tool_name="react_agent",
                metadata={"tools_used": result.get("intermediate_steps", [])}
            ))

            return self._create_query_response(result, query)

        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            return self._create_fallback_response(query, str(e))

    def _log_reasoning(self, steps: List[tuple]) -> None:
        """Track reasoning steps for transparency"""
        for step in steps:
            if isinstance(step[0], dict):
                thought = step[0].get("thought", "")
                action = step[0].get("action", "")
                observation = step[1]

                reasoning_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "thought": thought,
                    "action": action,
                    "observation": observation
                }

                self.reasoning_log.append(reasoning_entry)
                self.logger.debug(f"Reasoning step: {reasoning_entry}")


    def _create_fallback_response(self, query: str, error: str) -> QueryResponse:
        """Create graceful fallback response"""
        return QueryResponse(
            answer="I apologize, but I encountered an error processing your query. Please try rephrasing or ask another question.",
            citations=[],
            suggested_questions=[
                "Could you rephrase your question?",
                "Would you like to try a simpler query?",
                "Should we break this down into smaller questions?"
            ],
            metadata={
                "error": error,
                "original_query": query,
                "status": "failed"
            }
        )

    def _create_query_response(self, result: Dict[str, Any], query: str) -> QueryResponse:
        """Create QueryResponse from agent execution result"""
        try:
            # Extract intermediate steps properly
            steps = result.get("intermediate_steps", [])
            tool_outputs = [
                {
                    "tool": step[0].tool,
                    "input": step[0].tool_input,
                    "output": step[1]
                } for step in steps
            ]
            
            # Get tools by name from tool_map
            citation_tool = self.tool_map.get("citation_extractor")
            question_tool = self.tool_map.get("suggested_questions_generator")
            
            citations = []
            suggested_questions = []
            
            if citation_tool:
                citations = citation_tool.run({
                    "content": result["output"],
                    "tool_outputs": tool_outputs
                })
                
            if question_tool:
                suggested_questions = question_tool.run(
                    QuestionContext(
                        original_query=query,
                        answer_content=result["output"],
                        sources=tool_outputs,
                        metadata={"execution_steps": steps}
                    )
                )
                
            return QueryResponse(
                answer=result["output"],
                citations=citations,
                suggested_questions=suggested_questions,
                metadata={
                    "tools_used": [step[0].tool for step in steps],
                    "execution_time": result.get("execution_time"),
                    "tool_outputs": tool_outputs
                }
            )
        except Exception as e:
            self.logger.error(f"Error creating query response: {str(e)}")
            return self._create_fallback_response(query, str(e))
