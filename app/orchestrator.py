from .llm import LLMClient
from .config import settings
from .database import Database
from .agents.critic import Critic
from .retrieval.rag import RagIndex
from .agents.verifier import Verifier
from .agents.retriever import Retriever
from .retrieval.embedder import Embedder
from .agents.decomposer import Decomposer
from .agents.adjudicator import Adjudicator
from .schemas import Evidence, VerdictLabel
from .retrieval.learned_store import LearnedStore


def _merge_evidence(atual: list[Evidence], novas: list[Evidence]) -> list[Evidence]:
    vistos = {(e.source, e.snippet) for e in atual}
    for e in novas:
        chave = (e.source, e.snippet)
        if chave not in vistos:
            atual.append(e)
            vistos.add(chave)
    return atual


class Orchestrator:

    def __init__(self, db: Database) -> None:
        self.db = db
        self.llm = LLMClient()
        self.embedder = Embedder()
        self.critic = Critic(self.llm)
        self.rag = RagIndex(self.embedder)
        self.verifier = Verifier(self.llm)
        self.decomposer = Decomposer(self.llm)
        self.adjudicator = Adjudicator(self.llm)
        self.learned = LearnedStore(self.embedder, db)
        self.retriever = Retriever(self.llm, self.rag, self.learned)

    def warmup_sync(self) -> None:
        self.embedder.load()
        self.rag.load()

    async def load_learned(self) -> None:
        await self.learned.load()

    async def verify(self, claim: str) -> dict:
        claim_id = await self.db.insert_claim(claim)
        sub_claims = await self.decomposer.run(claim)

        resultados = []
        for sc in sub_claims:
            resultados.append(await self._verify_sub_claim(claim_id, sc))

        overall = self._aggregate(resultados)
        summary = self._build_summary(resultados, overall)
        await self.db.update_claim_verdict(claim_id, overall["label"], overall["confidence"], summary)

        return {
            "claim": claim,
            "summary": summary,
            "claim_id": claim_id,
            "sub_claims": resultados,
            "overall_label": overall["label"],
            "overall_confidence": overall["confidence"]
        }

    async def _verify_sub_claim(self, claim_id: int, sub_claim: str) -> dict:
        rodadas = []
        query = sub_claim
        verification = None
        evidencias: list[Evidence] = []
        max_extra = settings.max_debate_rounds

        for round_idx in range(max_extra + 1):
            novas = await self.retriever.run(query)
            evidencias = _merge_evidence(evidencias, novas)
            verification = await self.verifier.run(sub_claim, evidencias)
            critique = await self.critic.run(sub_claim, verification, evidencias)

            rodadas.append(
                {
                    "round": round_idx + 1,
                    "query": query,
                    "evidence": [e.model_dump() for e in novas],
                    "critique": critique.model_dump(mode="json"),
                    "verification": verification.model_dump(mode="json")
                }
            )
            await self.db.insert_round(
                query=query,
                claim_id=claim_id,
                sub_claim=sub_claim,
                round_idx=round_idx + 1,
                critique=critique,
                evidence=novas,
                verification=verification,
            )

            if not critique.has_objection or round_idx >= max_extra:
                break
            if critique.suggested_query:
                query = critique.suggested_query

        final = await self.adjudicator.run(sub_claim, verification)
        await self.db.insert_sub_claim(
            claim_id, sub_claim, final.label.value, final.confidence, final.rationale
        )

        return {
            "text": sub_claim,
            "label": final.label.value,
            "confidence": final.confidence,
            "rationale": final.rationale,
            "rounds": rodadas,
        }

    @staticmethod
    def _aggregate(resultados: list[dict]) -> dict:
        if not resultados:
            return {"label": VerdictLabel.NOT_ENOUGH_INFO.value, "confidence": 0.0}
        labels = [r["label"] for r in resultados]
        confs = [r["confidence"] for r in resultados]
        if any(l == "REFUTES" for l in labels):
            label = "REFUTES"
        elif all(l == "SUPPORTS" for l in labels):
            label = "SUPPORTS"
        else:
            label = "NOT_ENOUGH_INFO"
        return {"label": label, "confidence": round(sum(confs) / len(confs), 2)}

    @staticmethod
    def _build_summary(resultados: list[dict], overall: dict) -> str:
        n = len(resultados)
        sup = sum(1 for r in resultados if r["label"] == "SUPPORTS")
        ref = sum(1 for r in resultados if r["label"] == "REFUTES")
        nei = sum(1 for r in resultados if r["label"] == "NOT_ENOUGH_INFO")
        return (
            f"Veredito geral: {overall['label']} "
            f"(confiança {overall['confidence']:.0%}). "
            f"{n} sub-alegação(ões) analisada(s): {sup} sustentada(s), "
            f"{ref} refutada(s), {nei} sem evidência suficiente."
        )
