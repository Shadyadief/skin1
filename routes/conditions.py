"""
Conditions Routes — Knowledge Base + RAG Search
GET /api/conditions        — list all 10 conditions
GET /api/conditions/search — semantic RAG search
GET /api/conditions/:id    — get condition detail
"""

import logging
from fastapi import APIRouter, Query
from agents.rag_agent import rag_agent
from knowledge.dermnet_knowledge import DERMNET_CONDITIONS

logger = logging.getLogger("dermai.routes.conditions")
router = APIRouter()


@router.get("/conditions")
async def list_conditions():
    """Return all skin conditions from DermNet knowledge base."""
    return [
        {
            "id": c["id"],
            "name": c["name"],
            "category": c["category"],
            "icdCode": c.get("icd_code", ""),
            "description": c["description"],
            "symptoms": c.get("symptoms", []),
            "treatments": c.get("treatments", {}),
            "severity": c.get("severity_levels", [c.get("severity", "variable")]),
            "keywords": c.get("keywords", []),
        }
        for c in DERMNET_CONDITIONS
    ]


@router.get("/conditions/search")
async def search_conditions(q: str = Query(..., description="Search query")):
    """
    Semantic RAG search across knowledge base.
    Uses FAISS vector similarity (or keyword fallback).
    """
    if not q.strip():
        return {"results": [], "query": q, "retrieval_type": "none"}

    docs = rag_agent.retrieve(q, k=5, use_mmr=True)
    sources = rag_agent.get_sources_metadata(docs)

    results = []
    seen_conditions = set()
    for i, doc in enumerate(docs):
        condition_name = doc.metadata.get("condition_name", "General")
        if condition_name in seen_conditions:
            continue
        seen_conditions.add(condition_name)

        condition_data = next(
            (c for c in DERMNET_CONDITIONS if c["name"] == condition_name), None
        )

        results.append({
            "rank": len(results) + 1,
            "conditionName": condition_name,
            "category": doc.metadata.get("category", ""),
            "icdCode": doc.metadata.get("icd_code", ""),
            "chunkType": doc.metadata.get("chunk_type", ""),
            "relevantExcerpt": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,
            "conditionDetail": {
                "symptoms": condition_data.get("symptoms", []) if condition_data else [],
                "keywords": condition_data.get("keywords", []) if condition_data else [],
            } if condition_data else None
        })

    return {
        "query": q,
        "results": results,
        "totalFound": len(results),
        "retrievalType": rag_agent.embed_type or "keyword_fallback",
    }


@router.get("/conditions/{condition_id}")
async def get_condition(condition_id: int):
    """Get full detail for a single skin condition."""
    condition = next((c for c in DERMNET_CONDITIONS if c["id"] == condition_id), None)
    if not condition:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Condition not found")

    return {
        "id": condition["id"],
        "name": condition["name"],
        "category": condition["category"],
        "icdCode": condition.get("icd_code", ""),
        "description": condition["description"],
        "symptoms": condition.get("symptoms", []),
        "severityLevels": condition.get("severity_levels", []),
        "pathophysiology": condition.get("pathophysiology", ""),
        "treatments": condition.get("treatments", {}),
        "routine": condition.get("routine", {}),
        "lifestyle": condition.get("lifestyle", condition.get("triggers_to_avoid", [])),
        "ingredientsToAvoid": condition.get("ingredients_to_avoid", []),
        "expectedTimeline": condition.get("expected_timeline", "8-12 weeks"),
        "keywords": condition.get("keywords", []),
    }
