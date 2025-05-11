"""
Main class for the LLM setting.
"""
# External imports:
import os
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# Registry of supported LLM providers
LLM_PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "api_url_env": "OPENAI_URL",
        "model_env": "OPENAI_MODEL",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_url_env": "DEEPSEEK_URL",
        "model_env": "DEEPSEEK_MODEL",
    }

    # Add more providers here
}


class LLMClient:
    """
    Defines the client for sending API calls to LLM providers.
    """

    def __init__(self, system_prompt: str, user_prompt: str, temperature: float, provider: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature

        if provider not in LLM_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: '{provider}'")

        config = LLM_PROVIDERS[provider]
        self.api_key = os.getenv(config["api_key_env"])
        self.api_url = os.getenv(config["api_url_env"])
        self.model = os.getenv(config["model_env"])

        if not all([self.api_key, self.api_url, self.model]):
            raise EnvironmentError(f"Missing environment variables for provider '{provider}'")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self.data = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
            ]
        }

    def send_message(self) -> Optional[str]:
        """Send a synchronous request to the LLM and return only the assistant's reply."""
        try:
            response = requests.post(self.api_url, json=self.data, headers=self.headers, timeout=80)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.Timeout:
            print("Request timed out after 80 seconds.")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        return None

    def send_message_return_all(self) -> Optional[Dict[str, Any]]:
        """Send a synchronous request to the LLM and return full response JSON."""
        try:
            response = requests.post(self.api_url, json=self.data, headers=self.headers, timeout=80)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            print("Request timed out after 80 seconds.")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        return None
