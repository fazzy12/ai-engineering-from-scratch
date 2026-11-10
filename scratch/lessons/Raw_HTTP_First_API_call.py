import os
import urllib.request
import json

url = "https://api.anthropic.com/v1/messages"
headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
}

body = json.dumps({
    "model": os.environ.get("LLM_MODEL", "claude-3-5-sonnet-20241022"),
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "what is a neural network in one sentence?"}],
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method="POST")
try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(result["content"][0]["text"])
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    print("Response Body:", e.read().decode("utf-8"))