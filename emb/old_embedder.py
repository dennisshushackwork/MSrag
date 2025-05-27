# External imports:
import torch
import logging
from typing import List
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Initialising the Logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Embedder:
    """
    Embedding Service using the Singleton Design Pattern.
    Uses the Alibaba-NLP/gte-base-en-v.1.5 embedding model found on Huggingface.
    https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5
    """
    _instance = None # Holds the instance of the embedder
    _is_initialized = False # Flag for embedding initalisation

    def __new__(cls, model_id: str = "Alibaba-NLP/gte-base-en-v1.5"):
        """Creates a new embedder instance."""
        if cls._instance is None:
            logger.info("Creating new embedder instance.")
            cls._instance = super(Embedder, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_id: str = "Alibaba-NLP/gte-base-en-v1.5"):
        """Initialisation of the Embedding Class"""
        if not Embedder._is_initialized:
            logger.info("Initialising embedder instance with model ID '{}'".format(model_id))
            self.model_id = model_id
            self.max_tokens = 8192 # This is the maximum number of tokens the embedder can handle at once.
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load the model and the tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Load the model CPU or GPU and set the model to eval mode:
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
            self.model.eval()
            Embedder._is_initialized = True
            logger.info("Embedder initialization complete.")


    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts using the embedder and returns the embedding vectors."""
        if not texts:
            logger.warning("No texts to embed.")
            return []

        try:
            # Tokenize the input texts
            batch_dict = self.tokenizer(texts,
                                        max_length=self.max_tokens,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')

            # Ensure no gradients are computed during inference
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
            return embeddings_normalized.detach().numpy().tolist()
        except Exception as e:
            logger.error("An error occurred during the embedding: {}".format(e))
            raise

    def cosine_similarity(self, emb1:List[float], emb2:List[float]) -> float:
        """Calculates the cosine similarity between two embeddings."""
        emb_1 = torch.tensor(emb1, device=self.device)
        emb_2 = torch.tensor(emb2, device=self.device)

        # Compute cosine similarity
        similarity = torch.dot(emb_1, emb_2)
        return similarity.item()  # Convert tensor to scalar value

    def tokenize_to_ids(self, text: str) -> List[int]:
        """Converts the text into a list of IDs"""
        # The tokenizer will warn and truncate to its own model_max_length if self.max_tokens is larger.
        encoding = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_tokens,
            add_special_tokens=False  # Adds [CLS] and [SEP] tokens
        )
        return encoding['input_ids']

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Converts the token_ids into a list of strings"""
        return self.tokenizer.decode(token_ids, add_special_tokens=False)

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a text"""
        return len(self.tokenize_to_ids(text))


if __name__ == '__main__':
    document = """Hello my name is Dennis. I am a cool guy, looking for an improvement of my life.
                  Please make sure to include the best things into my name. What is the derivative of 5?
                  Born in Honolulu, Hawaii, Obama graduated from Columbia University in 1983 with a Bachelor of 
                  Arts degree in political science and later worked as a community organizer in Chicago. In 1988, 
                  Obama enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review.
                  He became a civil rights attorney and an academic, teaching constitutional law at the University 
                  of Chicago Law School from 1992 to 2004. In 1996, Obama was elected to represent the 13th district 
                  in the Illinois Senate, a position he held until 2004, when he successfully ran for the U.S. Senate. 
                  In the 2008 presidential election, after a close primary campaign against Hillary Clinton, he was 
                  nominated by the Democratic Party for president. Obama selected Joe Biden as his running mate and 
                  defeated Republican nominee John McCain and his running mate Sarah Palin."""
    query2 = "Dennis Shushack"
    query3 = "Colin Shushack"
    query4 = "Dennis Shushack"
    embedding = Embedder()
    emb1, emb2 = embedding.embed_texts([document, query2])
    print(len(emb1))
    print(embedding.cosine_similarity(emb1, emb2))
    tokens = embedding.tokenize_to_ids(document)
    print(tokens)
    print(embedding.decode_tokens(tokens))
    print(embedding.count_tokens(query3))





























