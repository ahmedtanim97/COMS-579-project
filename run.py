from fastapi import FastAPI
import gradio as gr

from gradio_ui import demo

app = FastAPI()

app = gr.mount_gradio_app(app,demo,path = '/gradio')

@app.get('/')
async def root():
    return app