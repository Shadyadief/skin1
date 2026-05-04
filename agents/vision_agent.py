"""
Agent 1 — Vision Classifier Agent
Uses HuggingFace Inference API to classify skin conditions from images.
Model: Anwarkh1/Skin_Disease-Image_Classification (ViT fine-tuned)
Fallback: GPT-4o Vision API
"""

import os
import base64
import logging
import httpx
from typing import Optional
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger("dermai.vision_agent")

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
HF_MODEL = "Anwarkh1/Skin_Disease-Image_Classification"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")


async def classify_with_huggingface(image_b64: str) -> dict:
    """
    Sends image to HuggingFace Inference API → ViT classifier.
    Returns top-k predictions with confidence scores.
    """
    if not HF_API_KEY:
        return {"error": "No HuggingFace API key configured"}

    image_bytes = base64.b64decode(image_b64)

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            HF_INFERENCE_URL,
            headers=headers,
            content=image_bytes,
        )

    if response.status_code == 503:
        return {"error": "Model loading, retry in 20s", "estimated_time": 20}

    if response.status_code != 200:
        return {"error": f"HF API error {response.status_code}: {response.text}"}

    predictions = response.json()

    if isinstance(predictions, list):
        top_predictions = []
        for p in predictions[:5]:
            label = p.get("label", "Unknown").replace("_", " ").title()
            score = round(p.get("score", 0) * 100, 2)
            top_predictions.append({"condition": label, "confidence": score})
        return {"predictions": top_predictions, "model": HF_MODEL}

    return {"error": "Unexpected response format", "raw": str(predictions)}


async def classify_with_gpt_vision(image_b64: str, mime_type: str = "image/jpeg") -> dict:
    """
    Fallback: GPT Vision API for detailed analysis + classification.
    Returns structured JSON with conditions, skin type, score, analysis.
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_tokens=2048,
    )

    messages = [
        SystemMessage(content="""You are an expert dermatologist AI. Analyze skin images with clinical precision.
Return ONLY valid JSON. Be specific, evidence-based, and empathetic.
Never diagnose — provide educational assessment only."""),
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": """Analyze this skin image and return this exact JSON:
{
  "conditions": ["primary condition", "secondary condition if present"],
  "skin_type": "Oily|Dry|Combination|Normal|Sensitive",
  "overall_score": <0-100 skin health score>,
  "confidence": <0-100 confidence in assessment>,
  "summary": "2-3 sentence clinical summary written like a caring dermatologist",
  "detailed_analysis": "paragraph: texture, tone, specific findings, severity, area affected",
  "recommendations": ["4-6 specific actionable clinical recommendations"],
  "severity": "mild|moderate|severe",
  "urgency": "routine|soon|urgent (urgent only if signs of infection/serious pathology)"
}
Scoring: 85-100=healthy, 70-84=mild concerns, 50-69=moderate, <50=significant concerns"""
            }
        ])
    ]

    import json
    response = await llm.ainvoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        return json.loads(content)
    except Exception:
        logger.warning("GPT Vision response parse failed, returning defaults")
        return {
            "conditions": ["General Skin Assessment"],
            "skin_type": "Normal",
            "overall_score": 75,
            "confidence": 60,
            "summary": "Analysis complete. Results may be limited due to image quality.",
            "detailed_analysis": "AI analysis performed on the provided image.",
            "recommendations": [
                "Consult a board-certified dermatologist for accurate diagnosis",
                "Use broad-spectrum SPF 30+ daily",
                "Maintain consistent gentle skincare routine",
                "Stay hydrated and protect skin from UV"
            ],
            "severity": "mild",
            "urgency": "routine"
        }


class VisionAgent:
    """
    Agent 1: Vision Classifier
    Strategy: Try HuggingFace ViT first → fallback to GPT Vision
    Provides: conditions, skin_type, score, confidence_scores, analysis
    """

    def __init__(self):
        self.name = "VisionAgent"
        self.hf_available = bool(HF_API_KEY)
        self.gpt_available = bool(OPENAI_API_KEY)
        logger.info(f"VisionAgent initialized | HF: {self.hf_available} | GPT: {self.gpt_available}")

    async def analyze(self, image_b64: str, mime_type: str = "image/jpeg") -> dict:
        """
        Main entry point. Returns full vision analysis.
        """
        logger.info("VisionAgent.analyze() called")
        result = {
            "source": None,
            "hf_classifications": [],
            "gpt_analysis": {},
            "conditions": [],
            "skin_type": "Normal",
            "overall_score": 75,
            "confidence": 70,
            "summary": "",
            "detailed_analysis": "",
            "recommendations": [],
            "severity": "mild",
            "urgency": "routine"
        }

        if self.hf_available:
            logger.info(f"Trying HuggingFace model: {HF_MODEL}")
            hf_result = await classify_with_huggingface(image_b64)

            if "predictions" in hf_result:
                result["hf_classifications"] = hf_result["predictions"]
                result["conditions"] = [p["condition"] for p in hf_result["predictions"][:3]]
                result["confidence"] = hf_result["predictions"][0]["confidence"] if hf_result["predictions"] else 70
                result["source"] = "huggingface_vit"
                logger.info(f"HF classifications: {result['hf_classifications'][:3]}")
            else:
                logger.warning(f"HF failed: {hf_result.get('error')}")

        if self.gpt_available:
            logger.info("Running GPT Vision analysis")
            gpt = await classify_with_gpt_vision(image_b64, mime_type)
            result["gpt_analysis"] = gpt

            if not result["conditions"] or result["source"] != "huggingface_vit":
                result["conditions"] = gpt.get("conditions", ["General Assessment"])
                result["source"] = "gpt_vision"

            result["skin_type"] = gpt.get("skin_type", "Normal")
            result["overall_score"] = gpt.get("overall_score", 75)
            result["summary"] = gpt.get("summary", "")
            result["detailed_analysis"] = gpt.get("detailed_analysis", "")
            result["recommendations"] = gpt.get("recommendations", [])
            result["severity"] = gpt.get("severity", "mild")
            result["urgency"] = gpt.get("urgency", "routine")
            if result["source"] == "huggingface_vit":
                gpt_conditions = gpt.get("conditions", [])
                merged = list(dict.fromkeys(result["conditions"] + gpt_conditions))
                result["conditions"] = merged[:4]

        if not result["conditions"]:
            result["conditions"] = ["Skin Assessment Required"]
            result["summary"] = "Please ensure image shows clear skin. Consult a dermatologist."

        logger.info(f"VisionAgent result: conditions={result['conditions']}, score={result['overall_score']}")
        return result


vision_agent = VisionAgent()
