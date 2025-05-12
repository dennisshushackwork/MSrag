"""
This class extracts all the entities from an input text.
"""

from llm.llm import LLMClient

class EntityExtractor:
    """Class designed to extract entities from a user query."""
    def __init__(self, query, model):
        self.query = query
        self.client = None
        self.model = model

    def system_prompt(self):
        return "You are an expert in extracting named entities from text."

    def human_prompt(self):

        return f"""
        Extract all named entities from the text below. 
       
        Input Text:
        {self.query}

        Instructions:
        - Identify and extract every named entity in the text.
        - Return the entities as a comma-separated list in LOWERCASE.
        - Do not include any additional text or explanations.
        - Do not answer a question if the text is a question.

        Example:
            Input:
            "Dennis Shushack is born in Hedingen and is a natural science student. He is 31 years old."
            Output:
            "dennis shushack,hedingen,natural science student,31 years"
        """

    def extract_entities(self):
        """Extract entities from a user query."""
        self.client = LLMClient(self.system_prompt(), self.human_prompt(), 1, provider=self.model)
        response = self.client.send_message()
        entities = response.split(",")
        query_entities = [entity.strip().strip('"').strip("'").lower() for entity in entities if entity.strip()]
        return query_entities

if __name__ == "__main__":
    extractor = EntityExtractor(query="""Barack Hussein Obama II[a] (born August 4, 1961) is an American politician who served as the 44th president of the United States
                       from 2009 to 2004""", model="openai")
    entities = extractor.extract_entities()
    print(entities)
