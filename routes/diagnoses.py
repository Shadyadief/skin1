"""
Diagnoses Routes — Full AI Pipeline endpoint
POST /api/diagnoses — runs 4-agent pipeline
GET  /api/diagnoses — list user diagnoses
GET  /api/diagnoses/:id — get diagnosis details
DELETE /api/diagnoses/:id — delete diagnosis
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from db.database import get_db, DiagnosisModel, ChatMessageModel
from agents.orchestrator import pipeline

logger = logging.getLogger("dermai.routes.diagnoses")
router = APIRouter()


def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    return x_user_id or "demo-user"


class CreateDiagnosisRequest(BaseModel):
    imageBase64: str
    mimeType: str = "image/jpeg"


@router.post("/diagnoses")
async def create_diagnosis(
    body: CreateDiagnosisRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    logger.info(f"New diagnosis request | user={user_id}")

    if not body.imageBase64:
        raise HTTPException(status_code=400, detail="imageBase64 is required")

    try:
        result = await pipeline.run_diagnosis(body.imageBase64, body.mimeType)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    image_url = f"data:{body.mimeType};base64,{body.imageBase64[:100]}..."

    diagnosis_id = -1

    if db is not None:
        try:
            diagnosis = DiagnosisModel(
                user_id=user_id,
                image_url=image_url,
                conditions=result["conditions"],
                skin_type=result["skin_type"],
                overall_score=result["overall_score"],
                confidence_scores=result.get("confidence_scores", []),
                summary=result["summary"],
                detailed_analysis=result.get("detailed_analysis", ""),
                recommendations=result.get("recommendations", []),
                rag_sources=result.get("rag_sources", []),
                routine=result.get("routine", {}),
                severity=result.get("severity", "mild"),
                urgency=result.get("urgency", "routine"),
                vision_source=result.get("vision_source", "unknown"),
                embed_type=result.get("embed_type", "keyword_fallback"),
            )
            db.add(diagnosis)
            db.flush()
            diagnosis_id = diagnosis.id

            welcome = await pipeline.build_welcome_message(diagnosis_id, result)
            chat_msg = ChatMessageModel(
                diagnosis_id=diagnosis_id,
                role="assistant",
                content=welcome,
                sources=result.get("rag_sources", [])
            )
            db.add(chat_msg)
            db.commit()
            db.refresh(diagnosis)
        except Exception as e:
            logger.error(f"DB error: {e}")
            db.rollback()
            diagnosis_id = -1
    else:
        diagnosis_id = 1

    return {
        "id": diagnosis_id,
        "userId": user_id,
        "imageUrl": image_url,
        "conditions": result["conditions"],
        "skinType": result["skin_type"],
        "overallScore": result["overall_score"],
        "confidenceScores": result.get("confidence_scores", []),
        "summary": result["summary"],
        "detailedAnalysis": result.get("detailed_analysis", ""),
        "recommendations": result.get("recommendations", []),
        "severity": result.get("severity", "mild"),
        "urgency": result.get("urgency", "routine"),
        "ragSources": result.get("rag_sources", []),
        "routine": result.get("routine", {}),
        "visionSource": result.get("vision_source", "unknown"),
        "embedType": result.get("embed_type", "keyword_fallback"),
        "createdAt": None,
    }


@router.get("/diagnoses")
def list_diagnoses(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    if db is None:
        return []

    try:
        diagnoses = (
            db.query(DiagnosisModel)
            .filter(DiagnosisModel.user_id == user_id)
            .order_by(DiagnosisModel.id.desc())
            .limit(50)
            .all()
        )
        return [
            {
                "id": d.id,
                "userId": d.user_id,
                "conditions": d.conditions,
                "skinType": d.skin_type,
                "overallScore": d.overall_score,
                "summary": d.summary,
                "severity": d.severity,
                "createdAt": d.created_at.isoformat() if d.created_at else None,
            }
            for d in diagnoses
        ]
    except Exception as e:
        logger.error(f"List diagnoses error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list diagnoses")


@router.get("/diagnoses/{diagnosis_id}")
def get_diagnosis(
    diagnosis_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    diagnosis = (
        db.query(DiagnosisModel)
        .filter(DiagnosisModel.id == diagnosis_id, DiagnosisModel.user_id == user_id)
        .first()
    )
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    messages = (
        db.query(ChatMessageModel)
        .filter(ChatMessageModel.diagnosis_id == diagnosis_id)
        .order_by(ChatMessageModel.id)
        .all()
    )

    return {
        "id": diagnosis.id,
        "userId": diagnosis.user_id,
        "imageUrl": diagnosis.image_url,
        "conditions": diagnosis.conditions,
        "skinType": diagnosis.skin_type,
        "overallScore": diagnosis.overall_score,
        "confidenceScores": diagnosis.confidence_scores,
        "summary": diagnosis.summary,
        "detailedAnalysis": diagnosis.detailed_analysis,
        "recommendations": diagnosis.recommendations,
        "ragSources": diagnosis.rag_sources,
        "routine": diagnosis.routine,
        "severity": diagnosis.severity,
        "urgency": diagnosis.urgency,
        "visionSource": diagnosis.vision_source,
        "chatHistory": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "createdAt": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
        "createdAt": diagnosis.created_at.isoformat() if diagnosis.created_at else None,
    }


@router.delete("/diagnoses/{diagnosis_id}", status_code=204)
def delete_diagnosis(
    diagnosis_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    diagnosis = (
        db.query(DiagnosisModel)
        .filter(DiagnosisModel.id == diagnosis_id, DiagnosisModel.user_id == user_id)
        .first()
    )
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    db.delete(diagnosis)
    db.commit()
