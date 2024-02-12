import requests

url = "http://127.0.0.1:5000/predict"
data = {"text1": "nuclear body seeks new tech", "text2": "terror suspects face arrest"}

response = requests.post(url, json=data)
print(response.json())
