import os
from openai import OpenAI

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                                api_key=os.environ["LLM_API_KEY"].strip())
for m in client.models.list().data:
    print(m.id)