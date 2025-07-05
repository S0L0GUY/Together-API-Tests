from together import Together
import os

client = Together(
    api_key=os.getenv('API_KEY')
)

with open("prompt.txt", 'r', encoding='utf-8') as mood_file:
    prompt = mood_file.read()

history = [{"role": "system", "content": prompt}]


while True:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=history
    )

    print(response.choices[0].message.content)

    history.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })

    user_input = input("You: ")
    history.append({
        "role": "user",
        "content": user_input
    })
