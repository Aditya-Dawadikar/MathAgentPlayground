from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent import build_agent
from app.models import ChatRequest, ChatResponse

load_dotenv()

_stateless_agent = None
_stateful_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stateless_agent, _stateful_agent
    _stateless_agent = build_agent(with_memory=False)
    _stateful_agent = build_agent(with_memory=True)
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
