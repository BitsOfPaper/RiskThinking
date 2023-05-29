# RiskThinking

Live prediction:
https://huggingface.co/spaces/BitsofPaper/Risk_Thinking


API:

pip install gradio_client

from gradio_client import Client

client = Client("https://bitsofpaper-risk-thinking.hf.space/")

result = client.predict("Volume Moving Avg", "Adj Close Rolling Mean", api_name="/predict")

print(result)


<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/3.32.0/gradio.js"
></script>

<gradio-app src="https://bitsofpaper-risk-thinking.hf.space"></gradio-app>
