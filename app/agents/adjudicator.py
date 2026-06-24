from .base import Agent
from ..schemas import AdjudicatorOutput, Verification

SYSTEM = (
    "Você dá a conclusão final sobre uma afirmação, em linguagem simples, para uma "
    "pessoa comum (sem conhecimento técnico). Decida se a afirmação é verdadeira, "
    "falsa ou se não foi possível confirmar, e escreva uma justificativa curta e clara.\n"
    "Regras OBRIGATÓRIAS para o campo 'rationale':\n"
    "- Escreva 1 ou 2 frases, em português simples, como se explicasse para um amigo.\n"
    "- É PROIBIDO usar os códigos SUPPORTS, REFUTES ou NOT_ENOUGH_INFO no texto.\n"
    "- É PROIBIDO usar termos do processo interno como 'sub-alegação', 'alegação', "
    "'veredito', 'rodada', 'debate', 'verificador' ou 'crítico'.\n"
    "- Não fale sobre como a análise foi feita; apenas explique POR QUE a afirmação é "
    "verdadeira, falsa ou não pôde ser confirmada.\n"
    "O campo 'label' é apenas um código interno e deve ser exatamente um de: "
    "SUPPORTS (verdadeira), REFUTES (falsa) ou NOT_ENOUGH_INFO (não confirmada). "
    "Calibração da 'confidence' (0.0 a 1.0):\n"
    "- NUNCA use 1.0; o máximo razoável é 0.9, pois sempre há incerteza.\n"
    "- Se as informações se contradizem ou o assunto é disputado/controverso, "
    "prefira NOT_ENOUGH_INFO com confiança BAIXA (0.2 a 0.4).\n"
    "- Só use confiança alta (acima de 0.8) quando a informação for clara, direta e "
    "vinda de fontes confiáveis, sem contradição.\n"
    'Responda APENAS em JSON: '
    '{"label": "SUPPORTS|REFUTES|NOT_ENOUGH_INFO", "confidence": 0.0, '
    '"rationale": "..."}'
)


class Adjudicator(Agent):

    async def run(self, sub_claim: str, verification: Verification) -> AdjudicatorOutput:
        user = (
            f"Afirmação: {sub_claim}\n"
            f"Avaliação prévia (código interno): {verification.label.value}\n"
            f"Resumo da avaliação: {verification.rationale}\n"
            "Dê a conclusão final. Lembre-se: a justificativa deve ser simples e sem "
            "termos técnicos, explicando por que a afirmação é verdadeira, falsa ou "
            "não pôde ser confirmada."
        )
        data = await self.llm.complete_json(SYSTEM, user)
        try:
            out = AdjudicatorOutput(**data)
        except Exception:
            return AdjudicatorOutput(
                confidence=0.5,
                label=verification.label,
                rationale=verification.rationale
            )

        out.confidence = min(out.confidence, 0.9)
        return out
