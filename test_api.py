import requests
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

response = requests.post(
    f"{API_URL}/predict",
    json={"TV": 100, "radio": 20, "newspaper": 10}
)
print(response.status_code, response.json())