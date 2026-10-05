import os
from openai import OpenAI

MODEL = "openai/gpt-oss-120b"     # use a name from your list if this one isn't there

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ["LLM_API_KEY"])
r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Say hello"}],
)
print(r.choices[0].message.content)