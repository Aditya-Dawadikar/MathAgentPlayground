from pydantic import BaseModel
from typing import Any, Optional


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None

class PlanRequest(BaseModel):
    problem: str


class ExecutionTraceRequest(BaseModel):
    problem: str


class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None


class ExecutionTraceResponse(BaseModel):
    plan: dict[str, Any]
    intermediate_results: list[Any]
    event_trace: list[dict[str, Any]]
    execution_trace: list[dict[str, Any]]
    final_output: Any


class ExecutionTraceFailureResponse(BaseModel):
    error: str
    step_number: int
    tool_name: str
    attempts: int
    reason: str
    intermediate_results: list[Any]
    event_trace: list[dict[str, Any]]
    execution_trace: list[dict[str, Any]]
