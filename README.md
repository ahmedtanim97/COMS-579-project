# COMS-579-project
Welcome to the COMS-579 Natural Language Processing project repository! This guide will help you set up your project environment and run the project scripts.

## 🚀 Setting Up Your Python Environment

### Creating new python environment

For macOS/Linux
```python<version> -m venv <virtual-environment-name>``` <br>

For Windows
```py -<version> -m venv <virtual-environment-name>```

## Activating environment
For macOS/Linux
```source <virtual-environment-name>/bin/activate``` <br>

For Windows<br>

modify the execution policy for PowerShell scripts (if needed) 
```Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass```<br>

Activate the environment
```<virtual-environment-name>\Scripts\activate```

## Requirements Installation
```pip install -r requirements.txt```

## Create a .env file in the root folder, follow the env-template and place your api_key inplace of the empty string with your openai and pinecone api key

### Command Line

#### How to upload and index PDF?

```python upload_and_index.py --pdf_file example.pdf```

 [Video](https://iastate.box.com/s/j2sklrpq6pagj847mw3wfosn461rzlxd)

#### Answer generation based on user query from command line

```python query.py --question="What is Mistral AI?"```

[Video](https://iastate.box.com/s/j9bf163h3wlj8i3xmg3tkeetisq93a47) 

##### Gradio UI command to run locally 

```python gradio_ui.py```

##### Gradio App deployment in Hugging Face Space
[Link](https://huggingface.co/spaces/ahmedtanim97/rag-system)

[Video](https://iastate.box.com/s/d3qw7su0h3nbjhfjn6tpmbn4izre02ql)