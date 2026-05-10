import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

DEFAULT_MODEL = "google/gemini-2.0-flash-001"


def call_llm(messages, model=DEFAULT_MODEL, max_tokens=256, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


def parse_label(raw_output, valid_labels):
    raw = raw_output.strip().lower()
    for label in valid_labels:
        if label.lower() in raw:
            return label
    return "unknown"
