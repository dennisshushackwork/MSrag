"""
Creates a response for the RAG-Pipeline.
"""
class RagPrompt:
    """Creates a RAG Response to the given user prompt."""
    def __init__(self, query: str, context: list[str]):
        self.query = query
        self.context = context
        self.system_prompt = self.system_prompt()
        self.human_prompt = self.human_prompt()

    @staticmethod
    def system_prompt() -> str:
        """Return a system prompt for RAG"""
        system_prompt = """You are an expert in Retrieval-Augmented Generation (RAG). Your task is to 
        answer user queries using provided context!"""
        return system_prompt

    def human_prompt(self) -> str:
        human_prompt = f"""
                      You have access to the following relevant document excerpts:
                      {self.context}

                      Given the user's question: '{self.query}'
                      Please provide the best possible answer, referencing only to the context if relevant.
                      Do not hallucinate. If the context is insufficient to answer the question reply with:
                      The context is insufficient to answer the question.
                      """
        return human_prompt

