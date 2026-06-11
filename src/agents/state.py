from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from typing import Annotated

from langgraph.graph.message import add_messages

class DietaryTrackerState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    intent: str
    extracted_items: List[Dict[str, str]]
    needs_clarification: bool
    clarification_question: str
    nutrition_data: Dict[str, Any]
    literature_context: str
    final_analysis: str
    error_logs: Optional[List[str]]
