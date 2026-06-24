from enum import Enum

from pydantic import BaseModel, Field


class VerdictLabel(str, Enum):
    REFUTES = "REFUTES"
    SUPPORTS = "SUPPORTS"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class Decomposition(BaseModel):
    sub_claims: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    title: str
    source: str
    snippet: str
    url: str | None = None
    age_days: int | None = None


class Verification(BaseModel):
    rationale: str
    label: VerdictLabel
    cited_snippet: str = ""


class Critique(BaseModel):
    has_objection: bool
    objection: str = ""
    suggested_query: str | None = None


class AdjudicatorOutput(BaseModel):
    rationale: str
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
