# RiskThinking

Live prediction:
https://huggingface.co/spaces/BitsofPaper/Risk_Thinking

--------------------------------------------------------------------------------------------
API:

pip install gradio_client

from gradio_client import Client

client = Client("https://bitsofpaper-risk-thinking.hf.space/")

result = client.predict("Volume Moving Avg", "Adj Close Rolling Mean", api_name="/predict")

print(result)

--------------------------------------------------------------------------------------------
App code:

https://huggingface.co/spaces/BitsofPaper/Risk_Thinking/tree/main


--------------------------------------------------------------------------------------------
* Make sure to set the Kaggle API credentials as environment variables in Dockerfile *
