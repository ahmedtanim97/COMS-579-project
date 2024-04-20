# COMS-579-project
COM S 579 NLP project

## Creating new python environment
```python<version> -m venv <virtual-environment-name>```
## Activating environment
```source <environment-name>/bin/activate```
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