from openai import OpenAI
import constants

OPENAI_API_KEY = constants.openai_key
client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the purpose of life?"}
    ],
    max_tokens=60
    )

print(completion.choices[0].message.content)

print("OPEN AI Connected Successfully")