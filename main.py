import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set in environment")


client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model=MODEL_NAME,
    input="Write a one-sentence bedtime story about a unicorn.",
    temperature=0.0
)

print(response)