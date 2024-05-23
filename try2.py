import ollama

print("What is the number 1 school in the Philippines")

response = ollama.chat(model='mistral', messages=[
  {
    'role': 'user',
    'content': 'What is the number 1 school in the Philippines?',
  },
])

print(response['message']['content'])
