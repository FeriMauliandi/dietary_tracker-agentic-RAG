from typing import TypedDict, List, Dict, Any, Optional

class DietaryTrackerState(TypedDict):
    user_input: str
    intent: str
    extracted_items: List[str]
    nutrition_data: Dict[str, Any]
    literature_context: str
    final_analysis: str
    error_logs: Optional[List[str]]