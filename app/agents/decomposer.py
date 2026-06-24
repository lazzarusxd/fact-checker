from .base import Agent
from ..config import settings
from ..schemas import Decomposition

SYSTEM = (
    "Você divide uma afirmação em fatos checáveis independentes, para que cada um "
    "seja verificado separadamente.\n"
    "REGRAS ABSOLUTAS:\n"
    "- NUNCA altere, corrija, reescreva ou 'conserte' a afirmação. Se ela estiver "
    "errada, NÃO é seu trabalho corrigir — apenas repasse como veio. Exemplo: se a "
    "afirmação for '2 + 2 é igual a 5', você devolve EXATAMENTE ['2 + 2 é igual a 5']; "
    "JAMAIS troque por '2 + 2 é igual a 4' nem crie fatos derivados.\n"
    "- NUNCA invente fatos que não estão escritos na afirmação.\n"
    "- Só divida se a afirmação contiver DOIS OU MAIS fatos realmente distintos "
    "(ligados por 'e', vírgulas, 'além de'). Cada item deve ser um pedaço FIEL do "
    "texto original.\n"
    "- Se for um único fato, devolva-o sozinho, exatamente como veio.\n"
    "- Nunca repita o mesmo fato com outras palavras.\n"
    f"- No máximo {settings.max_sub_claims} itens.\n"
    'Responda APENAS em JSON no formato: {"sub_claims": ["...", "..."]}'
)

_CONJ = (" e ", " além de ", " bem como ", ",", ";")


def _looks_compound(claim: str) -> bool:
    c = f" {claim.lower()} "
    return any(m in c for m in _CONJ)


def _dedup(subs: list[str]) -> list[str]:
    vistos, saida = set(), []
    for s in subs:
        chave = " ".join(s.lower().split())
        if chave and chave not in vistos:
            vistos.add(chave)
            saida.append(s)
    return saida


class Decomposer(Agent):

    async def run(self, claim: str) -> list[str]:
        if not _looks_compound(claim):
            return [claim]

        data = await self.llm.complete_json(SYSTEM, f"Afirmação: {claim}")
        try:
            subs = [s.strip() for s in Decomposition(**data).sub_claims if s.strip()]
        except Exception:
            subs = []
        subs = _dedup(subs) or [claim]
        return subs[: settings.max_sub_claims]
