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
    Scores are 1–10 per criterion; overall_score is the arithmetic mean (computed, not LLM-supplied).
    """

    understanding_needs: int = Field(
        ...,
        ge=1,
        le=10,
        description="Hiểu nhu cầu — discovery, listening, acknowledging pain.",
    )
    response_structure: int = Field(
        ...,
        ge=1,
        le=10,
        description="Cấu trúc trả lời — clarity, concision, professionalism.",
    )
    objection_handling: int = Field(
        ...,
        ge=1,
        le=10,
        description="Xử lý phản đối — empathy, reframe, non-defensive.",
    )
    persuasiveness: int = Field(
        ...,
        ge=1,
        le=10,
        description="Thuyết phục — value tied to the buyer's problems.",
    )
    next_steps: int = Field(
        ...,
        ge=1,
        le=10,
        description="Bước tiếp theo — close / micro-commitment.",
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
# Web App Schemas — Stateless endpoints for Next.js integration
# ================================================================

class MessageTurnSchema(BaseModel):
    """Single chat message (used in conversation_history)."""
    role: Literal["user", "assistant"]
    content: str


class WebChatRequest(BaseModel):
    """Request from Next.js — stateless, no SESSION_STORE."""
    message: str = Field(..., min_length=1, max_length=16000)
    conversation_history: list[MessageTurnSchema] = Field(default_factory=list)
    scenario_title: str = Field(default="Kịch bản luyện tập")
    scenario_description: str = Field(default="")
    customer_persona: str = Field(..., min_length=1, description="Full persona text from Supabase DB")
    company_context: str | None = None
    document_contents: str | None = Field(default=None, description="Nội dung tài liệu sản phẩm/quy trình bán hàng (text, concat từ nhiều file)")
    current_turn: int = Field(default=1, ge=1)
    max_turns: int = Field(default=12, ge=1, le=50)
    ai_tone: str = Field(default="neutral", description="Thái độ của AI: friendly, neutral, harsh")
    follow_up_depth: str = Field(default="moderate", description="Mức độ truy vấn: light, moderate, deep")
    time_remaining_seconds: int | None = None


class WebChatResponse(BaseModel):
    """Response to Next.js — structured customer reply."""
    customer_reply: str
    session_should_end: bool
    end_reason: str | None = None
    turn_count: int
    audio_url: str | None = Field(
        default=None, 
        description="Đường dẫn đến file âm thanh phản hồi từ FPT.AI TTS."
    )


class RubricScoreSchema(BaseModel):
    """Single rubric criterion score (0-10) + reason."""
    score: int = Field(..., ge=0, le=10)
    reason: str


class ImprovementSchema(BaseModel):
    """A sales sentence that needs improvement — verbatim from transcript."""
    user_sentence: str
    ai_suggestion: str
    playbook_source: str | None = Field(default=None, description="Trích nguồn từ tài liệu playbook/sản phẩm (nếu có)")


class WebRubricBreakdown(BaseModel):
    """6-criteria rubric matching Web App UI."""
    process_adherence: RubricScoreSchema
    talk_to_listen: RubricScoreSchema
    discovery_depth: RubricScoreSchema
    confidence: RubricScoreSchema
    objection_handling: RubricScoreSchema
    next_step: RubricScoreSchema


class WebEvaluateRequest(BaseModel):
    """Request from Next.js — Evaluate a completed session."""
    scenario_title: str = Field(default="Kịch bản luyện tập")
    scenario_description: str = Field(default="")
    customer_persona: str = Field(default="Khách hàng bình thường")
    document_contents: str | None = Field(default=None, description="Nội dung tài liệu sản phẩm/quy trình bán hàng")
    messages: list[MessageTurnSchema] = Field(..., min_length=1)


class WebEvaluateResponse(BaseModel):
    """Response to Next.js — 6-criteria evaluation + improvements + tips."""
    overall_score: float
    rubric_breakdown: WebRubricBreakdown
    improvements: list[ImprovementSchema] = Field(default_factory=list)
    top_3_tips: list[str] = Field(default_factory=list)


class WebGenerateScenarioRequest(BaseModel):
    """Request from Next.js to auto-generate scenario info from a document."""
    document_contents: str = Field(..., min_length=10, description="Nội dung file (đã chuyển thành text)")


class WebGenerateScenarioResponse(BaseModel):
    """Response with generated fields for the Scenario form."""
    title: str = Field(..., description="Tên kịch bản ngắn gọn, ấn tượng")
    description: str = Field(..., description="Mô tả bối cảnh để tập luyện")
    company_context: str = Field(..., description="Tóm tắt thông tin sản phẩm dùng làm background")
    customer_persona: str = Field(..., description="Một trong 3 ID: friendly_indecisive, detail_oriented, busy_skeptic")

