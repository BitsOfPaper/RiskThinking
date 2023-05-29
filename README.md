# RiskThinking

Live prediction:
https://huggingface.co/spaces/BitsofPaper/Risk_Thinking


API:

pip install gradio_client

from gradio_client import Client

client = Client("https://bitsofpaper-risk-thinking.hf.space/")

result = client.predict("Volume Moving Avg", "Adj Close Rolling Mean", api_name="/predict")

print(result)


<iframe
	src="https://bitsofpaper-risk-thinking.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
