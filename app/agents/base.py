from ..llm import LLMClient


class Agent:

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
