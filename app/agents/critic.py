from .base import Agent
from ..util import format_evidence
from ..schemas import Critique, Evidence, Verification

SYSTEM = (
    "Você é um crítico ADVERSARIAL. Seu trabalho é tentar DERRUBAR o veredito "
    "proposto, como um advogado da parte contrária. Verifique com rigor:\n"
    "- A evidência citada REALMENTE prova o que o veredito afirma, ou é só uma menção "
    "vaga / fora de contexto?\n"
    "- Existe alguma evidência na lista que CONTRADIZ o veredito? Se sim, ele ignorou?\n"
    "- O tema é DISPUTADO/controverso (versões oficiais e populares divergem)? Nesse "
    "caso, um veredito de 'verdadeiro' ou 'falso' é arriscado demais.\n"
    "- As fontes são confiáveis, ou o veredito se apoiou em redes sociais/blogs?\n\n"
    "Se houver QUALQUER fragilidade — contradição, fonte fraca, tema disputado ou "
    "salto lógico — levante objeção (has_objection=true) e proponha UMA nova consulta "
    "de busca (suggested_query) que ajude a encontrar a versão oficial ou a evidência "
    "contrária. Só marque has_objection=false se o veredito for sólido e sem "
    "contradições. "
    'Responda APENAS em JSON: {"has_objection": true|false, "objection": "...", '
    '"suggested_query": "..." | null}'
)


class Critic(Agent):

    async def run(self, sub_claim: str, verification: Verification, evidence: list[Evidence]) -> Critique:
        user = (
            f"Afirmação: {sub_claim}\n"
            f"Veredito proposto: {verification.label.value}\n"
            f"Justificativa: {verification.rationale}\n"
            f"Trecho citado: {verification.cited_snippet}\n\n"
            f"Evidências disponíveis:\n{format_evidence(evidence)}"
        )
        data = await self.llm.complete_json(SYSTEM, user, temperature=0.3)
        try:
            return Critique(**data)
        except Exception:
            return Critique(has_objection=False)
