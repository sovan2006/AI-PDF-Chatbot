import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE)


class LLMManager:

    def __init__(self):

        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. "
                "Add GOOGLE_API_KEY to your .env file."
            )

        self.model_name = "gemini-3.1-flash-lite"

        self.client = genai.Client(
            api_key=self.api_key
        )

        print(
            f"Initialized LLM Model: {self.model_name}"
        )

    def generate_response(self, prompt: str) -> str:

        try:

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            if not response.text:
                return "The model returned an empty response."

            return response.text.strip()

        except Exception as error:

            raise RuntimeError(
                f"Gemini generation failed: {error}"
            ) from error