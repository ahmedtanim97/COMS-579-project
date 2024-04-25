import gradio as gr
from upload_and_index import upload_and_index, query_index
from query import get_answer

def answer_question(question):
    return get_answer(question)

def index_pdf(file_path):
    upload_and_index(file_path.name)
    return "File indexed successfully."

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Enter your question")
            answer_output = gr.Text(label="Answer")
            question_button = gr.Button("Get Answer")
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF")
            pdf_output = gr.Text(label="Indexing Status")
            pdf_button = gr.Button("Index PDF")
    question_button.click(answer_question, inputs=question_input, outputs=answer_output)
    pdf_button.click(index_pdf, inputs=pdf_input, outputs=pdf_output)

demo.launch()
