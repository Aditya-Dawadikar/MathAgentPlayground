from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage

from agent_tool_call_correctness_v1 import build_agent
from agent_execution_trace_v1 import ExecutionStepFailure, build_agent as build_execution_trace_agent
from planner_agent_v1 import build_agent as build_planner_agent
from models import (
    ChatRequest,
    ChatResponse,
    ExecutionTraceFailureResponse,
    ExecutionTraceRequest,
    ExecutionTraceResponse,
    PlanRequest,
)

load_dotenv()

_stateless_agent = None
_stateful_agent = None
_stateless_planner_agent = None
_stateless_execution_trace_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stateless_agent, _stateful_agent, _stateless_planner_agent, _stateless_execution_trace_agent
    _stateless_agent = build_agent(with_memory=False)
    _stateful_agent = build_agent(with_memory=True)
    _stateless_planner_agent = build_planner_agent(with_memory=False)
    _stateless_execution_trace_agent = build_execution_trace_agent(with_memory=False)
    yield


app = FastAPI(title="LangGraph Agent API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, summary="Stateless single-turn chat")
async def chat(req: ChatRequest):
    """Send a message and get a response. No conversation history is retained."""
    messages = []
    if req.system_prompt:
        messages.append(SystemMessage(content=req.system_prompt))
    messages.append(HumanMessage(content=req.message))

    try:
        result = await _stateless_agent.ainvoke({"messages": messages})
        return ChatResponse(response=result["messages"][-1].content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponse,
    summary="Stateful multi-turn chat",
)
async def session_chat(session_id: str, req: ChatRequest):
    """Send a message within a named session. Conversation history is retained in memory."""
    messages = []
    if req.system_prompt:
        messages.append(SystemMessage(content=req.system_prompt))
    messages.append(HumanMessage(content=req.message))

    config = {"configurable": {"thread_id": session_id}}
    try:
        result = await _stateful_agent.ainvoke({"messages": messages}, config=config)
        return ChatResponse(
            response=result["messages"][-1].content,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan", response_model=ChatResponse, summary="Stateless single-turn chat")
async def plan(req: PlanRequest):
    """Send a message and get a response. No conversation history is retained."""
    messages = []
    if getattr(req, "system_prompt", None) is not None:
        messages.append(SystemMessage(content=req.system_prompt))
    messages.append(HumanMessage(content=req.problem))

    try:
        result = await _stateless_planner_agent.ainvoke({"messages": messages})
        return ChatResponse(response=result["messages"][-1].content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/execution-trace",
    response_model=ExecutionTraceResponse,
    summary="Run the planner and executor trace agent",
)
async def execution_trace(req: ExecutionTraceRequest):
    messages = [HumanMessage(content=req.problem)]

    try:
        result = await _stateless_execution_trace_agent.ainvoke({"messages": messages})
        return ExecutionTraceResponse(
            plan=result["plan"],
            intermediate_results=result["intermediate_results"],
            event_trace=result["event_trace"],
            execution_trace=result["execution_trace"],
            final_output=result["final_output"],
        )
    except ExecutionStepFailure as exc:
        failure = ExecutionTraceFailureResponse(
            error="execution_step_failed",
            step_number=exc.step_number,
            tool_name=exc.tool_name,
            attempts=exc.attempts,
            reason=exc.reason,
            intermediate_results=exc.intermediate_results,
            event_trace=exc.event_trace,
            execution_trace=exc.execution_trace,
        )
        return JSONResponse(status_code=422, content=failure.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
