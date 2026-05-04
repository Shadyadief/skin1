"""
Multi-Agent Orchestrator
Coordinates the 4 agents in a sequential pipeline:
  Agent 1 (Vision) → Agent 2 (RAG) → Agent 3 (Routine) → Agent 4 (Chat)

Each agent gets the output from previous agents as context.
"""

import logging
import base64
import json
from typing import Optional

from agents.vision_agent import vision_agent
from agents.rag_agent import rag_agent
from agents.routine_agent import routine_agent
from agents.chat_agent import chat_agent

logger = logging.getLogger("dermai.orchestrator")


class DermAIPipeline:
    """
    Full diagnosis pipeline:
    1. VisionAgent classifies image
    2. RAGAgent retrieves clinical knowledge for detected conditions
    3. RoutineAgent generates personalized routine using RAG context
    4. Result is stored in DB with RAG sources
    ChatAgent handles subsequent conversation (stateful, per diagnosis_id)
    """

    async def run_diagnosis(self, image_b64: str, mime_type: str = "image/jpeg") -> dict:
        """
        Full pipeline: image → complete diagnosis with routine.
        Returns structured result ready for DB insertion and API response.
        """
        logger.info("=== DermAI Pipeline START ===")

        logger.info("Step 1/3: Vision Analysis")
        vision_result = await vision_agent.analyze(image_b64, mime_type)
        conditions = vision_result.get("conditions", ["General Assessment"])
        skin_type = vision_result.get("skin_type", "Normal")
        overall_score = vision_result.get("overall_score", 75)
        severity = vision_result.get("severity", "mild")

        logger.info(f"Vision result: conditions={conditions}, score={overall_score}")

        logger.info("Step 2/3: RAG Knowledge Retrieval")
        retrieved_docs = rag_agent.retrieve_for_conditions(conditions)
        rag_context = rag_agent.format_context(retrieved_docs, max_chars=3500)
        rag_sources = rag_agent.get_sources_metadata(retrieved_docs)

        logger.info(f"RAG retrieved {len(retrieved_docs)} docs from {len(rag_sources)} sources")

        logger.info("Step 3/3: Routine Generation")
        routine = await routine_agent.generate(
            conditions=conditions,
            skin_type=skin_type,
            overall_score=overall_score,
            rag_context=rag_context,
            severity=severity
        )

        logger.info("=== DermAI Pipeline COMPLETE ===")

        return {
            "conditions": conditions,
            "skin_type": skin_type,
            "overall_score": overall_score,
            "confidence_scores": vision_result.get("hf_classifications", []),
            "summary": vision_result.get("summary", ""),
            "detailed_analysis": vision_result.get("detailed_analysis", ""),
            "recommendations": vision_result.get("recommendations", []),
            "severity": severity,
            "urgency": vision_result.get("urgency", "routine"),
            "rag_sources": rag_sources,
            "routine": routine,
            "vision_source": vision_result.get("source", "unknown"),
            "embed_type": rag_agent.embed_type or "keyword_fallback"
        }

    async def build_welcome_message(
        self, diagnosis_id: int, pipeline_result: dict
    ) -> str:
        """Generate contextual welcome message for chat from pipeline results."""
        conditions = pipeline_result["conditions"]
        skin_type = pipeline_result["skin_type"]
        score = pipeline_result["overall_score"]
        summary = pipeline_result.get("summary", "")
        sources = pipeline_result.get("rag_sources", [])
        severity = pipeline_result.get("severity", "mild")
        urgency = pipeline_result.get("urgency", "routine")
        vision_source = pipeline_result.get("vision_source", "AI")
        embed_type = pipeline_result.get("embed_type", "keyword_fallback")

        source_names = list(set(s["condition"] for s in sources[:3]))

        urgency_note = ""
        if urgency == "urgent":
            urgency_note = "\n\n⚠️ **Note:** Some findings may warrant prompt dermatologist evaluation. Please consult a professional soon."
        elif urgency == "soon":
            urgency_note = "\n\n📋 **Recommendation:** Consider scheduling a dermatologist consultation in the next few weeks."

        welcome = f"""Hello! I'm DermAI Pro, your AI-powered clinical skincare consultant. I've completed a comprehensive analysis of your skin. Here's what I found:

**🔬 Detected Conditions:** {', '.join(conditions)}
**💧 Skin Type:** {skin_type}
**📊 Skin Health Score:** {score}/100 ({'Excellent' if score >= 85 else 'Good' if score >= 70 else 'Moderate concerns' if score >= 50 else 'Significant concerns'})
**⚡ Severity:** {severity.title()}

{summary}

**📚 Knowledge Sources Used:** {', '.join(source_names) if source_names else 'DermNet Clinical Database'}
**🤖 Analysis by:** Vision AI ({vision_source}) + Semantic RAG ({embed_type})

I've created a **personalized evidence-based skincare routine** tailored to your specific conditions. Each step is grounded in clinical dermatology guidelines.{urgency_note}

Feel free to ask me anything about:
• Your specific skin conditions and their causes
• How to use the recommended products
• Ingredient interactions and what to avoid
• Lifestyle changes that can improve your skin
• When to see a dermatologist

What would you like to know first? 🌿"""

        return welcome


pipeline = DermAIPipeline()
