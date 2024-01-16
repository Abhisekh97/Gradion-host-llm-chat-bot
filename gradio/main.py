import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64
import requests, json
from config import CONFIG
from dotenv import load_dotenv, find_dotenv
import gradio as gr
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = CONFIG["HF_API_TOKEN"]
# Helper function

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=CONFIG['HF_API_SUMMARY_BASE']):
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))


def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']

gr.close_all()
demo = gr.Interface(fn=summarize, inputs="text", outputs="text")
demo.launch(share=True, server_port=CONFIG["port"])


# gr.close_all()
# def ner(input):
#     output = get_completion(input)
#     return {"text": input, "entities": output}

# gr.close_all()
# demo = gr.Interface(fn=ner,
#                     inputs=[gr.Textbox(label="Text to find entities", lines=2)],
#                     outputs=[gr.HighlightedText(label="Text with entities")],
#                     title="NER with dslim/bert-base-NER",
#                     description="Find entities using the `dslim/bert-base-NER` model under the hood!",
#                     allow_flagging="never",
#                     #Here we introduce a new tag, examples, easy to use examples for your application
#                     examples=["My name is Andrew and I live in California", "My name is Poli and work at HuggingFace"])
# demo.launch(share=True, server_port=int(CONFIG['port']))