from .base import Agent
from ..llm import LLMClient
from ..schemas import Evidence
from ..retrieval.rag import RagIndex
from ..retrieval.web_search import web_search
from ..retrieval.learned_store import LearnedStore


class Retriever(Agent):

    def __init__(self, llm: LLMClient, rag: RagIndex, learned: LearnedStore) -> None:
        super().__init__(llm)
        self.rag = rag
        self.learned = learned

    async def run(self, query: str) -> list[Evidence]:
        vistos: set[str] = set()
        evidencias: list[Evidence] = []

        def _add(ev: Evidence) -> None:
            if ev.snippet and ev.snippet not in vistos:
                evidencias.append(ev)
                vistos.add(ev.snippet)

        for r in await self.rag.search(query):
            _add(Evidence(source="rag", title=r["title"], snippet=r["snippet"]))

        for r in await self.learned.search(query):
            _add(
                Evidence(
                    url=r.get("url"),
                    source="learned",
                    title=r["title"],
                    snippet=r["snippet"],
                    age_days=r.get("age_days")
                )
            )

        consultas = [query, f"{query} é verdade ou mito? controvérsia"]
        for consulta in consultas:
            for r in await web_search(consulta):
                if r.get("snippet"):
                    _add(
                        Evidence(
                            url=r["url"],
                            source="web",
                            title=r["title"],
                            snippet=r["snippet"]
                        )
                    )
                    await self.learned.add(r["title"], r["snippet"], r["url"])

        return evidencias
