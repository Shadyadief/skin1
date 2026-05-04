"""
Chat Routes — SSE Streaming Chat with Agent 4
POST /api/diagnoses/:id/chat — stream AI response
GET  /api/diagnoses/:id/chat — get chat history
DELETE /api/diagnoses/:id/chat — clear chat session
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
from sqlalchemy.orm import Session

from db.database import get_db, DiagnosisModel, ChatMessageModel
from agents.chat_agent import chat_agent
from agents.rag_agent import rag_agent

logger = logging.getLogger("dermai.routes.chat")
router = APIRouter()


def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    return x_user_id or "demo-user"


class ChatRequest(BaseModel):
    message: str


async def sse_stream(
    message: str,
    diagnosis_id: int,
    diagnosis_data: dict,
    rag_context: str,
    db: Optional[Session],
) -> AsyncGenerator[str, None]:
    """SSE generator wrapping ChatAgent.stream_response(). Saves to DB after stream."""
    full_response = ""

    async for chunk_str in chat_agent.stream_response(
        user_message=message,
        diagnosis_id=diagnosis_id,
        diagnosis_data=diagnosis_data,
        rag_context=rag_context,
    ):
        try:
            chunk = json.loads(chunk_str)
            if "content" in chunk:
                full_response += chunk["content"]
            yield f"data: {chunk_str}\n\n"
        except Exception:
            yield f"data: {chunk_str}\n\n"

    if db is not None and full_response:
        try:
            user_msg = ChatMessageModel(
                diagnosis_id=diagnosis_id,
                role="user",
                content=message,
                sources=[]
            )
            db.add(user_msg)

            ai_msg = ChatMessageModel(
                diagnosis_id=diagnosis_id,
                role="assistant",
                content=full_response,
                sources=[]
            )
            db.add(ai_msg)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save chat messages: {e}")
            db.rollback()


@router.post("/diagnoses/{diagnosis_id}/chat")
async def stream_chat(
    diagnosis_id: int,
    body: ChatRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    diagnosis_data = {}

    if db is not None:
        try:
            diagnosis = (
                db.query(DiagnosisModel)
                .filter(
                    DiagnosisModel.id == diagnosis_id,
                    DiagnosisModel.user_id == user_id
                )
                .first()
            )
            if not diagnosis:
                raise HTTPException(status_code=404, detail="Diagnosis not found")

            diagnosis_data = {
                "conditions": diagnosis.conditions or [],
                "skin_type": diagnosis.skin_type or "Normal",
                "overall_score": diagnosis.overall_score or 75,
                "summary": diagnosis.summary or "",
                "severity": diagnosis.severity or "mild",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"DB fetch failed, using defaults: {e}")
            diagnosis_data = {
                "conditions": ["General Skin Assessment"],
                "skin_type": "Normal",
                "overall_score": 75,
                "summary": "Skin analysis completed.",
                "severity": "mild"
            }
    else:
        diagnosis_data = {
            "conditions": ["General Skin Assessment"],
            "skin_type": "Normal",
            "overall_score": 75,
            "summary": "AI analysis completed.",
            "severity": "mild"
        }

    conditions = diagnosis_data.get("conditions", [])
    retrieved_docs = rag_agent.retrieve_for_conditions(conditions)
    rag_context = rag_agent.format_context(retrieved_docs, max_chars=2000)

    return StreamingResponse(
        sse_stream(body.message, diagnosis_id, diagnosis_data, rag_context, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/diagnoses/{diagnosis_id}/chat")
def get_chat_history(
    diagnosis_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    if db is None:
        return []

    try:
        messages = (
            db.query(ChatMessageModel)
            .filter(ChatMessageModel.diagnosis_id == diagnosis_id)
            .order_by(ChatMessageModel.id)
            .all()
        )
        return [
            {
                "id": m.id,
                "diagnosisId": m.diagnosis_id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources or [],
                "createdAt": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ]
    except Exception as e:
        logger.error(f"Get chat history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat history")


@router.delete("/diagnoses/{diagnosis_id}/chat")
def clear_chat_history(
    diagnosis_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    chat_agent.clear_history(diagnosis_id)

    if db is not None:
        try:
            messages = (
                db.query(ChatMessageModel)
                .filter(ChatMessageModel.diagnosis_id == diagnosis_id)
                .all()
            )
            for msg in messages:
                db.delete(msg)
            db.commit()
        except Exception as e:
            logger.error(f"Clear chat error: {e}")
            db.rollback()

    return {"success": True, "message": "Chat history cleared"}
