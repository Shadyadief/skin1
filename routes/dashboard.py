"""
Dashboard Routes
GET /api/dashboard — aggregated stats for the user
"""

import logging
from fastapi import APIRouter, Depends, Header
from typing import Optional
from sqlalchemy.orm import Session

from db.database import get_db, DiagnosisModel
from agents.rag_agent import rag_agent

logger = logging.getLogger("dermai.routes.dashboard")
router = APIRouter()


def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    return x_user_id or "demo-user"


@router.get("/dashboard")
def get_dashboard(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    """
    Returns aggregated stats:
    - Total diagnoses count
    - Score trend over time
    - Top conditions frequency
    - RAG system status
    - Recent diagnoses
    """
    rag_status = {
        "vectorStore": rag_agent.vectorstore is not None,
        "embedType": rag_agent.embed_type or "not_initialized",
        "documentCount": len(rag_agent.documents),
        "indexLoaded": rag_agent.vectorstore is not None,
    }

    if db is None:
        return {
            "totalDiagnoses": 0,
            "recentDiagnoses": [],
            "conditionBreakdown": [],
            "scoreTrend": [],
            "topConditions": [],
            "ragStatus": rag_status,
            "message": "Database not configured — running in demo mode"
        }

    try:
        all_diagnoses = (
            db.query(DiagnosisModel)
            .filter(DiagnosisModel.user_id == user_id)
            .order_by(DiagnosisModel.id.desc())
            .all()
        )

        condition_counts: dict[str, int] = {}
        for d in all_diagnoses:
            for cond in (d.conditions or []):
                condition_counts[cond] = condition_counts.get(cond, 0) + 1

        condition_breakdown = sorted(
            [{"condition": k, "count": v} for k, v in condition_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )

        score_trend = [
            {
                "date": d.created_at.isoformat() if d.created_at else None,
                "score": d.overall_score,
                "conditions": d.conditions or []
            }
            for d in reversed(all_diagnoses[-30:])
        ]

        recent = all_diagnoses[:5]

        return {
            "totalDiagnoses": len(all_diagnoses),
            "recentDiagnoses": [
                {
                    "id": d.id,
                    "conditions": d.conditions,
                    "skinType": d.skin_type,
                    "overallScore": d.overall_score,
                    "summary": d.summary,
                    "severity": d.severity,
                    "createdAt": d.created_at.isoformat() if d.created_at else None,
                }
                for d in recent
            ],
            "conditionBreakdown": condition_breakdown,
            "scoreTrend": score_trend,
            "topConditions": [c["condition"] for c in condition_breakdown[:5]],
            "ragStatus": rag_status,
        }

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return {
            "totalDiagnoses": 0,
            "recentDiagnoses": [],
            "conditionBreakdown": [],
            "scoreTrend": [],
            "topConditions": [],
            "ragStatus": rag_status,
            "error": str(e)
        }
