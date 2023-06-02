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
* Make sure to set the Kaggle API credentials as environment variables in Dockerfile

* Build docker image and run conatiner.

* Output from running this container:

![rt](https://github.com/BitsOfPaper/RiskThinking/assets/83096475/65e0cd2f-1124-4391-9d5b-9af5be5acf17)

