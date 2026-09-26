from groq import Groq
from dotenv import load_dotenv
import os

# .env file se API key load karta hai
load_dotenv()

# Groq client banate hain
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Ek simple test message bhejte hain
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Say 'Setup successful!' if you can read this."}
    ]
)

# Result print karte hain
print(response.choices[0].message.content)