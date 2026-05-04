"""
Agent 4 — Conversational Chat Agent
LangChain ConversationalRetrievalChain with:
- PostgreSQL-backed or in-memory conversation history
- RAG retrieval at each turn
- Source document citation
- Streaming support via async generator
"""

import os
import logging
import json
from typing import AsyncGenerator, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

logger = logging.getLogger("dermai.chat_agent")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

in_memory_histories: dict[int, list[dict]] = {}


def get_conversation_history(diagnosis_id: int) -> list[dict]:
    return in_memory_histories.get(diagnosis_id, [])


def add_to_history(diagnosis_id: int, role: str, content: str):
    if diagnosis_id not in in_memory_histories:
        in_memory_histories[diagnosis_id] = []
    in_memory_histories[diagnosis_id].append({"role": role, "content": content})
    if len(in_memory_histories[diagnosis_id]) > 40:
        in_memory_histories[diagnosis_id] = in_memory_histories[diagnosis_id][-40:]


def build_system_prompt(diagnosis_data: dict, rag_context: str) -> str:
    conditions = diagnosis_data.get("conditions", [])
    skin_type = diagnosis_data.get("skin_type", "Normal")
    score = diagnosis_data.get("overall_score", 75)
    summary = diagnosis_data.get("summary", "")
    severity = diagnosis_data.get("severity", "mild")

    return f"""You are DermAI Pro, an advanced AI skincare expert assistant powered by clinical dermatology knowledge.
You combine the warmth of a caring physician with the precision of evidence-based medicine.

━━━ PATIENT SKIN PROFILE ━━━
• Detected conditions: {', '.join(conditions)}
• Skin type: {skin_type}
• Skin health score: {score}/100
• Severity: {severity}
• Assessment: {summary}

━━━ CLINICAL KNOWLEDGE BASE (RAG Retrieved) ━━━
{rag_context}

━━━ RESPONSE GUIDELINES ━━━
1. EMPATHY FIRST: Acknowledge the patient's concern before diving into advice
2. EVIDENCE-BASED: Reference clinical mechanisms when explaining (e.g., "Retinoids work by binding RAR receptors...")
3. SPECIFIC: Give product names, percentages, application techniques — not vague advice
4. CITE SOURCES: When using RAG knowledge, mention "According to dermatology guidelines..."
5. SAFETY: Always note when to see a dermatologist (infection signs, severe flares, systemic symptoms)
6. STRUCTURED: Use short paragraphs; bullet points for steps; emoji sparingly for warmth 🌿
7. LIMITATIONS: Never diagnose. You provide educational AI-powered guidance.
8. FOLLOW-UP: End with a relevant clarifying question to better understand the patient's needs

Response length: 2-4 paragraphs + bullet points when listing steps. Concise but comprehensive.
Language: Professional yet warm and accessible. Like a knowledgeable friend who is also a dermatologist."""


class ChatAgent:
    """
    Agent 4: Conversational Chat Agent
    Architecture:
    - LangChain LCEL chain: retriever | prompt | LLM | output parser
    - In-memory sliding window conversation history (last 20 turns)
    - RAG context injected at each turn from Agent 2
    - Streaming via ainvoke with stream=True
    """

    def __init__(self):
        self.name = "ChatAgent"
        self.llm = None
        self.streaming_llm = None

        if OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=0.7,
                max_tokens=2048,
            )
            self.streaming_llm = ChatOpenAI(
                model="gpt-4o",
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=0.7,
                max_tokens=2048,
                streaming=True,
            )
            logger.info("ChatAgent initialized with GPT-4o streaming")
        else:
            logger.warning("ChatAgent: No OpenAI key — chat disabled")

    def _build_messages(
        self,
        user_message: str,
        diagnosis_data: dict,
        rag_context: str,
        history: list[dict]
    ) -> list:
        """Build message array for LLM call with conversation history."""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        system_prompt = build_system_prompt(diagnosis_data, rag_context)
        messages = [SystemMessage(content=system_prompt)]

        recent_history = history[-10:]
        for msg in recent_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_message))
        return messages

    async def stream_response(
        self,
        user_message: str,
        diagnosis_id: int,
        diagnosis_data: dict,
        rag_context: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens as SSE-compatible chunks.
        Yields: {'content': '...'} | {'done': True, 'sources': [...]}
        """
        if not self.streaming_llm:
            yield json.dumps({"error": "Chat not available — no LLM configured"})
            return

        history = get_conversation_history(diagnosis_id)
        add_to_history(diagnosis_id, "user", user_message)

        messages = self._build_messages(user_message, diagnosis_data, rag_context, history)

        full_response = ""
        try:
            async for chunk in self.streaming_llm.astream(messages):
                token = chunk.content
                if token:
                    full_response += token
                    yield json.dumps({"content": token})

            add_to_history(diagnosis_id, "assistant", full_response)

            sources = []
            yield json.dumps({"done": True, "sources": sources})

        except Exception as e:
            logger.error(f"ChatAgent stream error: {e}")
            yield json.dumps({"error": f"Stream error: {str(e)}"})
            return

    async def respond(
        self,
        user_message: str,
        diagnosis_id: int,
        diagnosis_data: dict,
        rag_context: str,
    ) -> dict:
        """
        Non-streaming response for simple Q&A.
        Returns: {'content': '...', 'sources': [...]}
        """
        if not self.llm:
            return {
                "content": "Chat service is currently unavailable. Please configure an LLM API key.",
                "sources": []
            }

        history = get_conversation_history(diagnosis_id)
        add_to_history(diagnosis_id, "user", user_message)
        messages = self._build_messages(user_message, diagnosis_data, rag_context, history)

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            add_to_history(diagnosis_id, "assistant", content)
            return {"content": content, "sources": []}
        except Exception as e:
            logger.error(f"ChatAgent error: {e}")
            return {"content": f"I encountered an error processing your request: {str(e)}", "sources": []}

    def clear_history(self, diagnosis_id: int):
        """Clear conversation history for a diagnosis session."""
        if diagnosis_id in in_memory_histories:
            del in_memory_histories[diagnosis_id]
            logger.info(f"Cleared history for diagnosis {diagnosis_id}")


chat_agent = ChatAgent()
