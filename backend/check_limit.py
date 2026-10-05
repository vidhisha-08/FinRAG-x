import os
from openai import OpenAI
from config import MODEL

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ["LLM_API_KEY"].strip())
try:
    client.chat.completions.create(model=MODEL, max_tokens=5,
                                   messages=[{"role": "user", "content": "hi"}])
    print("OK: no limit right now")
except Exception as e:
    print(str(e))