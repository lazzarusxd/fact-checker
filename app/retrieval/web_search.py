import asyncio

from ddgs import DDGS

from ..config import settings

_BLOQUEADOS = (
    "tiktok.com", "instagram.com", "facebook.com", "pinterest.",
    "x.com", "twitter.com", "threads.net", "kwai.com",
)


def _confiavel(url: str) -> bool:
    u = (url or "").lower()
    return not any(dom in u for dom in _BLOQUEADOS)


def _search_sync(query: str, max_results: int) -> list[dict]:
    resultados: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results + 4):
                url = r.get("href", "")
                if not _confiavel(url):
                    continue
                resultados.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": url,
                    }
                )
                if len(resultados) >= max_results:
                    break
    except Exception as e:
        print(f"[web_search] aviso: {e}")
    return resultados


async def web_search(query: str, max_results: int | None = None) -> list[dict]:
    max_results = max_results or settings.web_results
    return await asyncio.to_thread(_search_sync, query, max_results)
