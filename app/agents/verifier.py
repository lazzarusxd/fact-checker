from .base import Agent
from ..util import format_evidence
from ..schemas import Evidence, Verification, VerdictLabel

SYSTEM = (
    "Você é um verificador factual CÉTICO e cuidadoso. Recebe uma afirmação e uma "
    "lista de evidências, e deve classificar como:\n"
    "- SUPPORTS: as evidências, de forma CLARA e CONSISTENTE, sustentam a afirmação;\n"
    "- REFUTES: as evidências, de forma clara e consistente, contradizem a afirmação;\n"
    "- NOT_ENOUGH_INFO: as evidências são insuficientes, irrelevantes, fracas OU se "
    "contradizem entre si (assunto disputado/controverso).\n\n"
    "Regras de julgamento:\n"
    "- Leia TODAS as evidências, inclusive as que vão CONTRA a afirmação. Se houver "
    "evidências relevantes em sentidos opostos, ou se o tema for reconhecidamente "
    "disputado/polêmico, escolha NOT_ENOUGH_INFO — não escolha um lado.\n"
    "- Pese a CONFIABILIDADE da fonte. Enciclopédias, órgãos oficiais e veículos "
    "jornalísticos valem muito mais que redes sociais, blogs ou opiniões. Uma simples "
    "MENÇÃO não é prova; desconfie de fontes fracas.\n"
    "- Para marcar SUPPORTS ou REFUTES, a evidência precisa ser direta e sem "
    "contradição relevante. Na dúvida, prefira NOT_ENOUGH_INFO.\n"
    "- ATENÇÃO a afirmações sobre RECONHECIMENTO OFICIAL (títulos, recordes, "
    "'campeão mundial', 'reconhecido pela FIFA' e similares). Para confirmá-las é "
    "preciso evidência DIRETA do órgão oficial responsável ou consenso claro entre "
    "fontes confiáveis. Termos como 'precursor', 'equivalente', 'intercontinental', "
    "'considerado por alguns', 'pela torcida', 'na prática' ou 'extraoficial' NÃO "
    "confirmam reconhecimento oficial — se o apoio à afirmação depende desses termos, "
    "escolha NOT_ENOUGH_INFO (ou REFUTES, se a afirmação disser que é oficial e as "
    "fontes mostrarem que o órgão NÃO reconhece).\n"
    "- Baseie-se SOMENTE nas evidências fornecidas, não no seu conhecimento prévio.\n"
    "- EXCEÇÃO: para fatos ELEMENTARES e universalmente aceitos (aritmética simples, "
    "lógica básica), você pode aplicar o conhecimento óbvio mesmo sem evidência. "
    "Ex.: '2 + 2 = 5' é claramente falso → REFUTES. Para fatos do mundo real "
    "(história, esporte, ciência aplicada, atualidades), continue se baseando nas "
    "evidências.\n"
    "- Cite o trecho exato da evidência que mais pesou na sua decisão.\n"
    'Responda APENAS em JSON: '
    '{"label": "SUPPORTS|REFUTES|NOT_ENOUGH_INFO", "rationale": "...", '
    '"cited_snippet": "..."}'
)


class Verifier(Agent):

    async def run(self, sub_claim: str, evidence: list[Evidence]) -> Verification:
        user = f"Afirmação: {sub_claim}\n\nEvidências:\n{format_evidence(evidence)}"
        data = await self.llm.complete_json(SYSTEM, user)
        try:
            return Verification(**data)
        except Exception:
            return Verification(
                label=VerdictLabel.NOT_ENOUGH_INFO,
                rationale="Não foi possível interpretar a resposta do verificador.",
            )
