"""
Pydantic request/response models for the Sale Train Agent FastAPI surface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class InitSessionRequest(BaseModel):
    scenario: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Scenario slug or label (e.g. cold_call_hesitation, negotiation).",
    )
    persona: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Buyer persona (e.g. skeptical, executive).",
    )
    max_turns: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Max rep turns before forced evaluation.",
    )


class InitSessionResponse(BaseModel):
    session_id: str
    scenario: str
    persona: str
    max_turns: int
    current_status: Literal["chatting", "evaluating", "completed"]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(
        ...,
        min_length=1,
        max_length=16000,
        description="Learner (sales rep) message for this turn.",
    )


class ChatResponse(BaseModel):
    session_id: str
    customer_reply: str
    current_status: Literal["chatting", "evaluating", "completed"]
    turn_count: int
    graph_trace: list[str] = Field(default_factory=list)
    session_ended: bool = Field(
        ...,
        description="True when this invocation finished the session (evaluation ran).",
    )
    end_reason: str | None = None


class EvaluationPayload(BaseModel):
    """
    Strict evaluation rubric for _evaluation_node / GET /feedback.
    Scores are 1â€“10 per criterion; overall_score is the arithmetic mean (computed, not LLM-supplied).
    """

    understanding_needs: int = Field(
        ...,
        ge=1,
        le=10,
        description="Hiá»ƒu nhu cáº§u â€” discovery, listening, acknowledging pain.",
    )
    response_structure: int = Field(
        ...,
        ge=1,
        le=10,
        description="Cáº¥u trÃºc tráº£ lá»i â€” clarity, concision, professionalism.",
    )
    objection_handling: int = Field(
        ...,
        ge=1,
        le=10,
        description="Xá»­ lÃ½ pháº£n Ä‘á»‘i â€” empathy, reframe, non-defensive.",
    )
    persuasiveness: int = Field(
        ...,
        ge=1,
        le=10,
        description="Thuyáº¿t phá»¥c â€” value tied to the buyer's problems.",
    )
    next_steps: int = Field(
        ...,
        ge=1,
        le=10,
        description="BÆ°á»›c tiáº¿p theo â€” close / micro-commitment.",
    )

    strengths: list[str] = Field(
        default_factory=list,
        description="What the learner did well.",
    )
    key_mistakes: list[str] = Field(
        default_factory=list,
        description="Critical errors or missed opportunities.",
    )
    suggested_better_answer: str = Field(
        ...,
        min_length=1,
        description="How a senior rep would respond in this exact context.",
    )

    @computed_field
    @property
    def overall_score(self) -> float:
        total = (
            self.understanding_needs
            + self.response_structure
            + self.objection_handling
            + self.persuasiveness
            + self.next_steps
        )
        return round(total / 5.0, 2)


class FeedbackResponse(BaseModel):
    session_id: str
    current_status: Literal["chatting", "evaluating", "completed"]
    evaluation_results: EvaluationPayload | None = None


class RetryResponse(BaseModel):
    session_id: str
    scenario: str
    persona: str
    max_turns: int
    current_status: Literal["chatting", "evaluating", "completed"]
    message: str = "Session reset. Same scenario and persona. Good luck!"


class NextLevelResponse(BaseModel):
    session_id: str
    scenario: str
    old_persona: str
    new_persona: str
    max_turns: int
    current_status: Literal["chatting", "evaluating", "completed"]
    message: str


# ================================================================
# Web App Schemas â€” Stateless endpoints for Next.js integration
# ================================================================

class MessageTurnSchema(BaseModel):
    """Single chat message (used in conversation_history)."""
    role: Literal["user", "assistant"]
    content: str


class WebChatRequest(BaseModel):
    """Request from Next.js â€” stateless, no SESSION_STORE."""
    message: str = Field(..., min_length=1, max_length=16000)
    llm_provider: str | None = Field(default=None, description="auto, openrouter, gpt, gemini, grok")
    llm_model: str | None = Field(default=None, description="Override model for this request")
    conversation_history: list[MessageTurnSchema] = Field(default_factory=list)
    scenario_title: str = Field(default="Ká»‹ch báº£n luyá»‡n táº­p")
    scenario_description: str = Field(default="")
    customer_persona: str = Field(..., min_length=1, description="Full persona text from Supabase DB")
    company_context: str | None = None
    document_contents: str | None = Field(default=None, description="Ná»™i dung tÃ i liá»‡u sáº£n pháº©m/quy trÃ¬nh bÃ¡n hÃ ng (text, concat tá»« nhiá»u file)")
    current_turn: int = Field(default=1, ge=1)
    max_turns: int = Field(default=12, ge=1, le=50)
    ai_tone: str = Field(default="neutral", description="ThÃ¡i Ä‘á»™ cá»§a AI: friendly, neutral, harsh")
    follow_up_depth: str = Field(default="moderate", description="Má»©c Ä‘á»™ truy váº¥n: light, moderate, deep")
    time_remaining_seconds: int | None = None


class WebChatResponse(BaseModel):
    """Response to Next.js â€” structured customer reply."""
    customer_reply: str
    session_should_end: bool
    end_reason: str | None = None
    turn_count: int
    audio_url: str | None = Field(
        default=None, 
        description="ÄÆ°á»ng dáº«n Ä‘áº¿n file Ã¢m thanh pháº£n há»“i tá»« FPT.AI TTS."
    )


class RubricScoreSchema(BaseModel):
    """Single rubric criterion score (0-10) + reason."""
    score: int = Field(..., ge=0, le=10)
    reason: str


class ImprovementSchema(BaseModel):
    """A sales sentence that needs improvement â€” verbatim from transcript."""
    user_sentence: str
    ai_suggestion: str
    playbook_source: str | None = Field(default=None, description="TrÃ­ch nguá»“n tá»« tÃ i liá»‡u playbook/sáº£n pháº©m (náº¿u cÃ³)")


class WebRubricBreakdown(BaseModel):
    """6-criteria rubric matching Web App UI."""
    process_adherence: RubricScoreSchema
    talk_to_listen: RubricScoreSchema
    discovery_depth: RubricScoreSchema
    confidence: RubricScoreSchema
    objection_handling: RubricScoreSchema
    next_step: RubricScoreSchema


class WebEvaluateRequest(BaseModel):
    llm_provider: str | None = Field(default=None, description="auto, openrouter, gpt, gemini, grok")
    llm_model: str | None = Field(default=None, description="Override model for this request")
    """Request from Next.js â€” Evaluate a completed session."""
    scenario_title: str = Field(default="Ká»‹ch báº£n luyá»‡n táº­p")
    scenario_description: str = Field(default="")
    customer_persona: str = Field(default="KhÃ¡ch hÃ ng bÃ¬nh thÆ°á»ng")
    document_contents: str | None = Field(default=None, description="Ná»™i dung tÃ i liá»‡u sáº£n pháº©m/quy trÃ¬nh bÃ¡n hÃ ng")
    messages: list[MessageTurnSchema] = Field(..., min_length=1)


class WebEvaluateResponse(BaseModel):
    """Response to Next.js â€” 6-criteria evaluation + improvements + tips."""
    overall_score: float
    rubric_breakdown: WebRubricBreakdown
    improvements: list[ImprovementSchema] = Field(default_factory=list)
    top_3_tips: list[str] = Field(default_factory=list)


class WebGenerateScenarioRequest(BaseModel):
    llm_provider: str | None = Field(default=None, description="auto, openrouter, gpt, gemini, grok")
    llm_model: str | None = Field(default=None, description="Override model for this request")
    """Request from Next.js to auto-generate scenario info from a document."""
    document_contents: str = Field(..., min_length=10, description="Ná»™i dung file (Ä‘Ã£ chuyá»ƒn thÃ nh text)")


class WebGenerateScenarioResponse(BaseModel):
    """Response with generated fields for the Scenario form."""
    title: str = Field(..., description="TÃªn ká»‹ch báº£n ngáº¯n gá»n, áº¥n tÆ°á»£ng")
    description: str = Field(..., description="MÃ´ táº£ bá»‘i cáº£nh Ä‘á»ƒ táº­p luyá»‡n")
    company_context: str = Field(..., description="TÃ³m táº¯t thÃ´ng tin sáº£n pháº©m dÃ¹ng lÃ m background")
    customer_persona: str = Field(..., description="Má»™t trong 3 ID: friendly_indecisive, detail_oriented, busy_skeptic")

