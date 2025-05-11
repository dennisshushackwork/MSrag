"""
This class is designed to extract all entities and relationships from the given
Text chunk using a Large Language Model.
"""
# External Imports:
from typing import Dict, Any, List

# Internal Imports:
from llm.llm import LLMClient

class Extractor:
    """Entities & Relationship Extractor with LLM"""
    def __init__(self, chunk: str, model: str):
        self.chunk = chunk
        self.model = model
        self.temperature = 1.0
        self.client = None

    @staticmethod
    def get_system_prompt() -> str:
        """Returns the system prompt for the LLM"""
        system_prompt = """You are an expert LLM in extracting triplets (entity-relationship-entity) from 
               text to create a Knowledge Graph for Retrieval Augmentation."""
        return system_prompt

    def get_user_prompt(self) -> str:
        """Returns the user prompts for the LLM call (extraction call)"""
        prompt = f"""
                
                ## Goal:
                Extract the relationship tuples from a given text chunk.
                
                ## Instructions:
                First analyze the given text and identify key entities and their relationships.
                Format each relationship as: source|target|description|summary. 
    
                Where:
                    - source and target are key entities extracted (lowercase)
                    - description is a complete sentence describing the relationship
                    - summary is a concise verb phrase (e.g., works_for, located_in)
                    
                ### Guidelines:
                1.Identify and extract all key entities (i.e., people, organizations, locations)
                2.Extract relationships between pairs of IDENTIFIED KEY entities.
                3.Include both explicit and strongly implied relationships
                4.Keep ONLY the key relationships essential for understanding the text.
                5.Integrate contextual details (dates, locations) into the relationships
                    - I.e. do not create a seperate entity for a date/time but rather integrate it into a relationship.
                6.Avoid creating separate relationships for minor details
                7.For tables: treat each row as source + header/value pairs
                8.For key-value lists: make keys predicates and values targets
                9.Make sure to include casual relationships (e.g., because of, since etc.)
                
                ### Formatting guidelines:
                - Return ONLY the relationships in the given format: source|target|description|summary
                - Separate each relationship with a newline
                - Use lowercase for all text
                - Use | to separate the fields
                - Do not include any additional text or explanations
                - Do not include any punctuation or special characters
                - Do not include duplicate relationships

                ### Example:
                Input Text: 
                The European Union threatened to renew a tariff on chamois leather from China for another five years to curb import competition
                for U.K. producers. The EU said it would review whether to let lapse the 58.9 percent duty on the soft leather, which is used to dry cars and windows. 
                The bloc imposed the levy in 2006 to punish Chinese exporters of chamois leather for selling it in Europe below cost, a practice known as dumping.
                The goal was to protect EU producers such as U.K.-based Hutchings & Harding Ltd. from less expensive imports. 
                The review will determine whether the expiry of the levy “would be likely, or unlikely, to lead to a continuation of dumping and injury,” 
                the European Commission, the 27-nation EU’s trade authority in Brussels, said today in the Official Journal. The duty was due to expire this week 
                and will now stay in place during the probe, which can last as long as 15 months. The inquiry results from a June 14 request by the U.K. Leather Federation, 
                which represents more than half of EU production of chamois leather, according to the commission. When imposing the anti-dumping duty five years ago, 
                the EU said Chinese exporters tripled their share of the European market to 30 percent in the 12 months to end-March 2005 compared with 2001.

                Expected Output:
                european union|china|the european union threatened to renew a tariff on chamois leather from china for another five years|imposes_tariff_on
                european union|uk producers|the european union threatened tariffs to curb import competition for uk producers|protects
                european union|chamois leather|the european union reviews 58.9 percent duty on chamois leather imports|regulates_import_of
                chamois leather|cars|chamois leather is used to dry cars|used_for
                chamois leather|windows|chamois leather is used to dry windows|used_for
                eu|chinese exporters|the eu imposed the levy in 2006 to punish chinese exporters for dumping|penalizes_for_dumping
                chinese exporters|europe|chinese exporters sold chamois leather in europe below cost|practiced_dumping_in
                hutchings & harding ltd|uk|hutchings & harding ltd is a uk-based chamois leather producer|based_in
                european commission|brussels|the european commission is the 27-nation eu's trade authority in brussels|headquartered_in
                duty|probe|the 58.9 percent duty will stay in place during the probe which can last up to 15 months|continues_during
                uk leather federation|inquiry|the uk leather federation requested the inquiry on june 14|initiated
                uk leather federation|eu production|the uk leather federation represents more than half of eu production of chamois leather|represents
                chinese exporters|european market|chinese exporters tripled their share of the european market to 30 percent by march 2005|increased_share_in
                european commission|official journal|the european commission announced the review in the official journal|communicates_through
                european union|duty expiry|the european union will determine if duty expiry would lead to continued dumping and injury|evaluates_impact_of
                
                Text to extract relationship from: {self.chunk}
                """
        return prompt

    def parse_format(self, content: str) -> Dict[str, Any]:
        """Parses the format (source|target|description|summary) into relationships."""
        relationships = []
        content = content.strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        for i, line in enumerate(lines):
            try:
                parts = line.split('|')

                if len(parts) == 4:
                    # Get the different parts of the extracted relationship:
                    source = parts[0].strip().lower()
                    target = parts[1].strip().lower()
                    relationship_description = parts[2].strip().lower()
                    relationship_summary = parts[3].strip().lower()

                    if source and target and relationship_description and relationship_summary:
                        relationships.append({
                            "source": source,
                            "target": target,
                            "description": relationship_description,
                            "summary": relationship_summary,
                        })
                    else:
                        print(f"Warning: Skipped line {i+1} due to missing content in one or more fields after stripping: '{line}'")
                else:
                    print(f"Warning: Skipped line {i+1} due to unexpected number of parts ({len(parts)} instead of 4): '{line}'")
            except Exception as e:
                print(f"Error parsing line {i+1}: '{line}'. Error: {str(e)}")
                continue

        return {"relationships": relationships}

    def call(self):
        """Calls the LLM without extracting the relationships"""
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt()
        self.client = LLMClient(user_prompt, system_prompt, self.temperature, provider=self.model)
        response = self.client.send_message_return_all()
        return response

    def call_and_extract(self) -> Dict[str, Any]:
        """Calls the LLM and extracts the relationships"""
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt()

        # Create client based on type
        self.client = LLMClient(user_prompt, system_prompt, self.temperature, provider=self.model)

        # Send message and get response
        response = self.client.send_message()
        # Parse the response
        extracted_data = self.parse_format(response)
        return extracted_data


if __name__ == "__main__":
    chunk = """During  a  high-level  security  meeting  in  Washington,  D.C.,  intelligence  officials  discussed  the  details  of  Operation  Neptune's  Spear,  sometimes  referred  to  as  Operation  Neptune's  Spear.  This  historic  mission,  coordinated  from  Washington,  DC,  was  later  acknowledged  by  several  U.S.  presidents,  including  George  H.  W.  Bush  and  his  son,  George  W.  Bush,  both  of  whom  played  pivotal  roles  in  shaping  U.S.  foreign  policy.  In  the  same  meeting  room  in  Washington,  D.C.,  where  many  former  presidents  had  once  made  decisions,  Vice  President  Kamala  Harris  reviewed  updates  from  the  intelligence  briefings.  Her  leadership  continues  a  legacy  shaped  by  progressive  policies  like  the  Don't  Ask,  Don't  Tell  Repeal  Act  of  2010,  which  is  sometimes  simply  cited  as  the  Don't  Ask,  Don't  Tell  Repeal  Act.  Social  reforms  were  also  top  of  mind,  especially  with  respect  to  same-sex  marriage,  a  topic  that  had  long  been  debated  under  the  broader  umbrella  of  same-sex  marriages  in  America.  One  document  reviewed  mentioned  the  American  Recovery  and  Reinvestment  Act,  while  another  spelled  out  the  American  Recovery  and  Reinvestment  Act  of  2009-both  referring  to  the  same  stimulus  initiative.  Meanwhile,  in  the  financial  world,  figures  like  Robert  S.  Kapito  (also  listed  as  Robert  Kapito)  of  BlackRock,  Inc.-or  simply  BlackRock  Inc.  in  some  filings-have  increasingly  faced  pressure  from  shareholder  votes  and  activists  scrutinizing  shareholder  voting  patterns.  Much  of  the  scrutiny  centers  around  environmental,  social  and  corporate  governance  (ESG)  policies,  sometimes  also  phrased  as  environmental,  social,  and  governance  (ESG).  Media  coverage  of  these  developments  earned  recognition  at  the  Primetime  Emmy  Awards,  also  referenced  in  some  captions  as  the  Primetime  Emmy  Award.  Notably,  Maya  Soetoro,  also  known  as  Maya  Soetoro-Ng,  weighed  in  on  social  justice  reforms  like  the  expansion  of  federal  hate  crime  law,  or  more  comprehensively,  federal  hate  crime  laws. """
    extractor = Extractor(chunk, model="openai")
    extractor_response = extractor.call_and_extract()
    print(extractor_response["relationships"])



