"""
Agent 3 — Routine Generator Agent
Uses LangChain 1.x create_agent (tool-calling) pattern.
Tools: ingredient compatibility, ingredient info, routine order validation.
Fallback: Direct LLM call if agent fails.
"""

import os
import json
import logging
from typing import Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("dermai.routine_agent")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

INCOMPATIBLE_PAIRS = [
    ("retinol", "aha", "Retinol + AHA: over-exfoliation risk. Use on alternate nights."),
    ("retinol", "bha", "Retinol + BHA: irritation. Alternate nights."),
    ("vitamin c", "niacinamide", "High-dose L-AA + niacinamide: apply 15 min apart."),
    ("benzoyl peroxide", "retinol", "BP oxidizes retinol. BP in AM, retinol in PM."),
    ("aha", "bha", "Multiple acids: risk of over-exfoliation. Alternate AM/PM."),
    ("vitamin c", "retinol", "Both potent actives. Vitamin C AM, retinol PM."),
]

INGREDIENT_DATABASE = {
    "salicylic acid": {
        "mechanism": "BHA — lipid-soluble, penetrates follicles, dissolves sebum plugs, anti-inflammatory via prostaglandin inhibition",
        "best_for": ["acne", "oily skin", "blackheads", "enlarged pores"],
        "avoid_if": ["aspirin allergy", "pregnancy (high doses)"],
        "concentration": "0.5-2% OTC",
        "ph_optimal": "3-4"
    },
    "niacinamide": {
        "mechanism": "Inhibits melanosome transfer; downregulates PPARγ sebocyte differentiation; ceramide synthesis stimulation; NF-κB anti-inflammatory",
        "best_for": ["hyperpigmentation", "oily skin", "barrier repair", "aging"],
        "avoid_if": ["niacin flush >10%"],
        "concentration": "4-10%",
        "ph_optimal": "5-7"
    },
    "hyaluronic acid": {
        "mechanism": "Hygroscopic: binds 1000x weight in water. Multi-MW for surface + epidermal hydration",
        "best_for": ["dry skin", "all skin types", "dehydration"],
        "avoid_if": ["very dry climate without occlusive on top"],
        "concentration": "0.1-2% multi-MW preferred",
        "ph_optimal": "5-8"
    },
    "retinol": {
        "mechanism": "Provitamin A → retinoic acid → RAR nuclear receptors → collagen synthesis, MMP inhibition, cell turnover, sebaceous gland reduction",
        "best_for": ["aging", "acne", "hyperpigmentation", "oily skin"],
        "avoid_if": ["pregnancy", "eczema flare", "rosacea", "post-procedure"],
        "concentration": "0.025-0.1% beginners, 0.3-1% experienced",
        "ph_optimal": "Neutral"
    },
    "vitamin c": {
        "mechanism": "Antioxidant (ROS neutralization), collagen synthesis co-factor, tyrosinase inhibitor, photoprotection",
        "best_for": ["hyperpigmentation", "aging", "dull skin", "sun damage"],
        "avoid_if": ["irritated barrier"],
        "concentration": "10-20% L-AA at pH 2.5-3.5",
        "ph_optimal": "2.5-3.5"
    },
    "azelaic acid": {
        "mechanism": "Tyrosinase inhibitor + anti-inflammatory (neutrophil ROS) + antibacterial (C. acnes) + antikeratinizing",
        "best_for": ["acne", "rosacea", "hyperpigmentation", "melasma", "sensitive skin"],
        "avoid_if": ["none significant"],
        "concentration": "10-15% OTC, 15-20% Rx",
        "ph_optimal": "4.0-4.5"
    },
    "ceramides": {
        "mechanism": "Barrier lipids (ceramide NP/AP/EOP + cholesterol + fatty acids 3:1:1) restore lamellar body structure, reduce TEWL",
        "best_for": ["dry skin", "eczema", "barrier repair", "sensitive skin", "rosacea"],
        "avoid_if": ["none"],
        "concentration": "formulation-dependent",
        "ph_optimal": "5-6"
    },
    "glycolic acid": {
        "mechanism": "AHA (smallest MW, deepest penetration): accelerates corneocyte desquamation, stimulates collagen I/III synthesis, pigment dispersion",
        "best_for": ["hyperpigmentation", "aging", "acne", "texture"],
        "avoid_if": ["sensitive skin", "rosacea", "same night as retinol"],
        "concentration": "5-10% OTC, up to 70% professional",
        "ph_optimal": "3.5-4.0"
    }
}


@tool
def check_ingredient_compatibility(ingredients: str) -> str:
    """
    Check if skincare ingredients are compatible. 
    Input: comma-separated ingredient names like 'retinol, glycolic acid, niacinamide'.
    Returns compatibility warnings and recommendations.
    """
    ingredient_list = [i.strip().lower() for i in ingredients.split(",")]
    warnings = []
    safe_notes = []

    for ing1, ing2, warning in INCOMPATIBLE_PAIRS:
        ing1_found = any(ing1 in ing for ing in ingredient_list)
        ing2_found = any(ing2 in ing for ing in ingredient_list)
        if ing1_found and ing2_found:
            warnings.append(f"⚠️ {warning}")

    if not warnings:
        safe_notes.append("✅ Ingredients are compatible for same-session use.")

    for ing in ingredient_list:
        for db_key, db_info in INGREDIENT_DATABASE.items():
            if db_key in ing or ing in db_key:
                safe_notes.append(f"ℹ️ {db_key.title()}: {db_info['mechanism'][:80]}...")
                break

    return "\n".join(warnings + safe_notes) or "No specific compatibility data found."


@tool
def get_ingredient_info(ingredient_name: str) -> str:
    """
    Get clinical details about a skincare ingredient.
    Input: ingredient name like 'niacinamide', 'retinol', 'glycolic acid'.
    Returns mechanism, concentration, best uses, contraindications.
    """
    name_lower = ingredient_name.lower().strip()
    for db_key, info in INGREDIENT_DATABASE.items():
        if db_key in name_lower or name_lower in db_key:
            return (
                f"INGREDIENT: {db_key.title()}\n"
                f"MECHANISM: {info['mechanism']}\n"
                f"BEST FOR: {', '.join(info['best_for'])}\n"
                f"AVOID IF: {', '.join(info['avoid_if'])}\n"
                f"CONCENTRATION: {info['concentration']}\n"
                f"OPTIMAL pH: {info['ph_optimal']}"
            )
    return f"No detailed clinical data for '{ingredient_name}'."


@tool
def validate_routine_layering(routine_steps_json: str) -> str:
    """
    Validate skincare routine step ordering (thinnest to thickest rule).
    Input: JSON string like '[{"step": "Moisturizer", "order": 1}, {"step": "Serum", "order": 2}]'.
    Returns validation result with any reordering issues.
    """
    correct_sequence = [
        "cleanser", "toner", "essence", "serum", "eye cream",
        "treatment", "moisturizer", "oil", "sunscreen", "spf"
    ]
    try:
        steps = json.loads(routine_steps_json)
    except Exception:
        return "Could not parse steps JSON. Proceeding with standard ordering."

    issues = []
    for i in range(1, len(steps)):
        prev_name = steps[i-1].get("step", "").lower()
        curr_name = steps[i].get("step", "").lower()
        prev_idx = next((j for j, s in enumerate(correct_sequence) if s in prev_name), 99)
        curr_idx = next((j for j, s in enumerate(correct_sequence) if s in curr_name), 99)
        if curr_idx < prev_idx:
            issues.append(
                f"⚠️ Reorder: '{steps[i].get('step')}' should come before '{steps[i-1].get('step')}'"
            )

    return "\n".join(issues) if issues else "✅ Routine layering order is correct."


class RoutineAgent:
    """
    Agent 3: Personalized Routine Generator
    LangChain 1.x: create_agent (tool-calling) pattern
    Tools: compatibility check, ingredient info, layering validation
    Fallback: direct LLM call
    """

    def __init__(self):
        self.name = "RoutineAgent"
        self.llm = None
        self.tools = [check_ingredient_compatibility, get_ingredient_info, validate_routine_layering]

        if OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=0.3,
                max_tokens=4096,
            )
            logger.info("RoutineAgent initialized with GPT-4o + 3 tools")
        else:
            logger.warning("RoutineAgent: No OpenAI key — limited functionality")

    async def generate(
        self,
        conditions: list[str],
        skin_type: str,
        overall_score: int,
        rag_context: str,
        severity: str = "mild"
    ) -> dict:
        if not self.llm:
            return self._default_routine(conditions)

        try:
            return await self._run_tool_calling_agent(conditions, skin_type, overall_score, rag_context, severity)
        except Exception as e:
            logger.warning(f"Tool-calling agent failed: {e}, using direct LLM")
            return await self._direct_llm_generate(conditions, skin_type, overall_score, rag_context)

    async def _run_tool_calling_agent(
        self,
        conditions: list[str],
        skin_type: str,
        score: int,
        rag_context: str,
        severity: str
    ) -> dict:
        """
        LangChain 1.x tool-calling pattern:
        LLM.bind_tools(tools) → LCEL chain → astream_events
        """
        from langchain.agents import create_agent

        system_prompt = f"""You are an expert clinical skincare formulator.
Create a personalized, evidence-based skincare routine using your tools.

PATIENT PROFILE:
- Conditions: {', '.join(conditions)}
- Skin type: {skin_type}  
- Score: {score}/100
- Severity: {severity}

CLINICAL KNOWLEDGE:
{rag_context[:2500]}

WORKFLOW:
1. Call check_ingredient_compatibility with your planned key ingredients
2. Call get_ingredient_info for any ingredients needing clarification
3. Call validate_routine_layering with your proposed morning steps JSON
4. Return the complete routine as valid JSON

REQUIRED OUTPUT FORMAT (valid JSON only):
{{
  "morning": [
    {{"order": 1, "step": "Cleanser", "product": "CeraVe Foaming", "active": "ceramides", "instruction": "60s massage, lukewarm water", "why": "clinical reason"}}
  ],
  "evening": [...],
  "weekly": [...],
  "key_ingredients": ["ingredient1"],
  "ingredient_warnings": ["warning if any"],
  "expected_timeline": "X weeks",
  "lifestyle_tips": ["tip1", "tip2"]
}}

Requirements: 5+ morning steps, 4+ evening steps, 2+ weekly steps. Real product names."""

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=f"Generate routine for {skin_type} skin with: {', '.join(conditions)}")]}
        )

        messages = result.get("messages", [])
        final_content = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                final_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if "{" in final_content:
            json_start = final_content.find("{")
            json_end = final_content.rfind("}") + 1
            parsed = json.loads(final_content[json_start:json_end])
            logger.info("RoutineAgent (tool-calling) generated routine successfully")
            return parsed

        raise ValueError("No valid JSON in agent output")

    async def _direct_llm_generate(
        self,
        conditions: list[str],
        skin_type: str,
        score: int,
        rag_context: str
    ) -> dict:
        """Direct LLM call without agent loop (fast fallback)."""
        messages = [
            SystemMessage(content=f"""You are an expert dermatology consultant.
Create a precise personalized skincare routine. Return ONLY valid JSON.

Clinical knowledge base:
{rag_context[:2000]}"""),
            HumanMessage(content=f"""
Patient profile: {skin_type} skin | Conditions: {', '.join(conditions)} | Score: {score}/100

Return this JSON structure with real product names:
{{
  "morning": [
    {{"order": 1, "step": "Cleanser", "product": "Specific product name", "active": "key ingredient", "instruction": "exact how-to", "why": "clinical reason"}},
    {{"order": 2, "step": "Serum/Treatment", "product": "...", "active": "...", "instruction": "...", "why": "..."}},
    {{"order": 3, "step": "Moisturizer", "product": "...", "active": "...", "instruction": "...", "why": "..."}},
    {{"order": 4, "step": "Sunscreen", "product": "SPF 30-50 specific", "active": "zinc oxide/chemical", "instruction": "2mg/cm²", "why": "UV protection"}}
  ],
  "evening": [
    {{"order": 1, "step": "Cleanser", ...}},
    {{"order": 2, "step": "Active Treatment", ...}},
    {{"order": 3, "step": "Moisturizer", ...}}
  ],
  "weekly": [
    {{"order": 1, "step": "Exfoliation", ...}},
    {{"order": 2, "step": "Mask", ...}}
  ],
  "key_ingredients": ["ceramides", "niacinamide"],
  "ingredient_warnings": [],
  "expected_timeline": "8-12 weeks for visible improvement",
  "lifestyle_tips": ["Change pillowcase 2x/week", "Drink 8 glasses water daily"]
}}

Include 5+ morning, 4+ evening, 2+ weekly steps. Tailor specifically to {', '.join(conditions)}.""")
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(content)
        except Exception as e:
            logger.error(f"Direct LLM parse failed: {e}")
            return self._default_routine(conditions)

    def _default_routine(self, conditions: list[str]) -> dict:
        return {
            "morning": [
                {"order": 1, "step": "Cleanser", "product": "CeraVe Foaming Facial Cleanser", "active": "ceramides, niacinamide", "instruction": "Gentle 60s massage with lukewarm water. Pat dry.", "why": "Removes overnight oils without stripping skin barrier"},
                {"order": 2, "step": "Serum", "product": "The Ordinary Niacinamide 10% + Zinc 1%", "active": "niacinamide, zinc gluconate", "instruction": "2-3 drops pressed into skin after toner", "why": "Sebum regulation, pore minimizing, barrier support"},
                {"order": 3, "step": "Moisturizer", "product": "CeraVe Moisturizing Cream", "active": "ceramides NP/AP/EOP, HA, cholesterol", "instruction": "Pea-sized, apply to slightly damp skin", "why": "Barrier restoration and hydration lock"},
                {"order": 4, "step": "Sunscreen", "product": "EltaMD UV Clear SPF 46", "active": "zinc oxide 9%, niacinamide", "instruction": "Quarter teaspoon (2mg/cm²). Final step. Reapply every 2h outdoors.", "why": "Non-negotiable UV protection — prevents condition worsening"}
            ],
            "evening": [
                {"order": 1, "step": "Cleanser", "product": "CeraVe Foaming Facial Cleanser", "active": "ceramides", "instruction": "Double cleanse if wearing SPF/makeup. Micellar water first.", "why": "Remove all daytime accumulation"},
                {"order": 2, "step": "Active Treatment", "product": "The Ordinary Niacinamide 10% + Zinc 1%", "active": "niacinamide", "instruction": "2-3 drops, gentle press-in motion", "why": "Condition-targeted treatment"},
                {"order": 3, "step": "Moisturizer", "product": "CeraVe PM Facial Moisturizing Lotion", "active": "ceramides, hyaluronic acid, niacinamide", "instruction": "Final step. Apply generously.", "why": "Overnight barrier repair and hydration"}
            ],
            "weekly": [
                {"order": 1, "step": "Exfoliation", "product": "The Ordinary Glycolic Acid 7% Toning Solution", "active": "glycolic acid 7%", "instruction": "Cotton pad, 1x/week. Do not rinse. Follow with moisturizer.", "why": "Cell turnover, texture refinement, pigmentation"},
                {"order": 2, "step": "Hydrating Mask", "product": "COSRX Advanced Snail 96 Mucin Mask", "active": "snail secretion filtrate 96%", "instruction": "20 min, 2x/week. Pat remaining essence in.", "why": "Deep hydration and barrier repair"}
            ],
            "key_ingredients": ["ceramides", "niacinamide", "zinc"],
            "ingredient_warnings": [],
            "expected_timeline": "8-12 weeks for visible improvement",
            "lifestyle_tips": [
                "Patch test all new products on inner arm for 48h",
                "Introduce one new product at a time",
                "Use SPF every day, even indoors (UV penetrates windows)"
            ]
        }


routine_agent = RoutineAgent()
