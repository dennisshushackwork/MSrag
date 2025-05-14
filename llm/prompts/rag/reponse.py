"""
Creates a response for the RAG-Pipeline.
"""
from llm.llm import LLMClient

class RagPrompt:
    """Creates a RAG Response to the given user prompt."""
    def __init__(self, query: str, context: list[str], model:str):
        self.query = query
        self.context = context
        self.client = None
        self.model = model

    @staticmethod
    def system_prompt() -> str:
        """Return a system prompt for RAG"""
        system_prompt = """You are an expert in Retrieval-Augmented Generation (RAG). Your task is to generate an optimal 
        answer for a user query, given a provided context!"""
        return system_prompt

    def human_prompt(self) -> str:
        human_prompt = f"""
                    You are a helpful knowledge assistant tasked with answering a user's question based solely on the provided context.
                    USER QUESTION: {self.query}
                    
                    CONTEXT:
                    {self.context}
                    
                    INSTRUCTIONS:
                    1. Carefully analyze the user's question and the provided context.
                    2. Construct a comprehensive, accurate response using ONLY information found in the context.
                    3. Format your answer as complete sentences in a conversational but informative tone.
                    4. If the question involves multiple parts, address each part systematically.
                    5. If specific details requested are not found in the context, acknowledge this limitation rather than making assumptions.
                    6. Do not reference external knowledge or include information absent from the context.
                    7. If the context is entirely insufficient to provide a meaningful answer to the question, return an empty response.
                    
                    ANSWER:
                    """
        return human_prompt

    def generate_response(self):
        """Extract entities from a user query."""
        print(self.human_prompt())
        self.client = LLMClient(self.system_prompt(), self.human_prompt(), 1, provider=self.model)
        response = self.client.send_message()
        return response


