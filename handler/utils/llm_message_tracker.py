from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

from config.common_settings import CommonConfig
from utils.logging_util import logger


@dataclass
class LLMInteraction:
    timestamp: datetime
    user_id: str
    session_id: str
    request_id: str
    prompt: str
    response: str
    tool_name: str
    metadata: Dict[str, Any]


class LLMMessageTracker:
    def __init__(self, config: CommonConfig):
        self.interactions: List[LLMInteraction] = []
        self.logger = logger

    def track(self, interaction: LLMInteraction):
        self.interactions.append(interaction)
        self.logger.debug(f"Tracked LLM interaction: {interaction}")
