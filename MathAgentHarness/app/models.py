from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None

class PlanRequest(BaseModel):
    problem: str


class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
