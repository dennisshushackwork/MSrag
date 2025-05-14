# Internal imports:
from llm.llm import LLMClient


class RelationshipAgent:
    """Class designed to extract entities from a user query."""
    def __init__(self, user_query, relationships, model):
        self.query = user_query
        self.relationships = relationships
        self.model = model
        self.client = None

    def system_prompt(self):
        """Gives us the system prompt."""
        return "You are an expert in evaluating IF a given CONTEXT (in form of sentences) is sufficient to ANSWER a user query."

    def human_prompt(self):
        return f"""
               ## Instructions:
               You are provided with the following user query: {self.query} and
               given the provided context: {self.relationships}. 
               
               Evaluate if the provided context is sufficient to ANSWER a user query.
               If the context is sufficient ANSWER with YES, else answer with NO.

               ### Output FORMAT:
               - ONLY RETURN THE WORD YES OR NO
               - DO NOT INCLUDE ANY OTHER TEXT 

               ## Example:
               User query: Who was Barack Obama married to?
               Context: Barack Obama is the 44th president, Obama lives in the White House.
               Expected Answer: NO
               """

    def check_context(self):
        """Extract entities from a user query."""
        self.client = LLMClient(self.system_prompt(), self.human_prompt(), 1, provider=self.model)
        response = self.client.send_message()
        return response

