import requests

headers = {
    "Authorization": f"Bearer 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"
}

response = requests.post(
    "https://api-inference.huggingface.co/models/openai-community/gpt2",
    headers=headers,
    json={"inputs": "Tell me a short joke about Python."}
)

print("Status code:", response.status_code)

try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Error decoding response:", response.text)
print("here")
exit()