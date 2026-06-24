from .schemas import Evidence


def format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(nenhuma evidência encontrada)"
    blocos = []
    for i, e in enumerate(evidence, 1):
        origem = "BASE LOCAL" if e.source == "rag" else "WEB"
        blocos.append(f"[{i}] ({origem}) {e.title}\n{e.snippet}")
    return "\n\n".join(blocos)
