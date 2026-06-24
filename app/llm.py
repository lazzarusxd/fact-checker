import json

from openai import AsyncOpenAI

from .config import settings


class LLMClient:

    def __init__(self) -> None:
        self._model = settings.llm_model
        self._client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url
        )

    async def complete_json(self, system: str, user: str, temperature: float = 0.2) -> dict:
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            response_format={"type": "json_object"},  # type: ignore
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]  # type: ignore
        )
        content = resp.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(content[start : end + 1])
            raise
