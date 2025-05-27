# Internal imports:
from llm.llm import LLMClient

class CommunitySummary:
    """Class Designed to create community summaries"""

    def __init__(self, relationships, model):
        self.relationships = relationships
        self.model = model
        self.client = None

    def system_prompt(self):
        """Gives us the system prompt."""
        return ("You are an expert AI assistant skilled in summarizing complex information. "
                "Your task is to generate a concise and coherent summary of a community based on a list "
                "of relationships describing connections between entities within that community. "
                "The relationships are provided as a list of textual descriptions.")

    def human_prompt(self):
        return f"""
                Please generate a concise summary with a suitable title for a community based on the following internal relationships:
               
                {self.relationships}  
                
                The summary should capture the main themes, key entities, and types of interactions observed within this community.
                Focus on what these relationships collectively reveal about the community's nature or purpose.
                The summary should be a paragraph (few sentences long), providing a clear and informative overview.
                Do not mention the word community within the summary.            
               """

    def generate_summary(self):
        """Generates the summary."""
        self.client = LLMClient(self.system_prompt(), self.human_prompt(), 0.1 , provider=self.model)
        response = self.client.send_message()
        return response

