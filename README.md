# Dating chatbot API

A Python chatbot service that provides a REST API.

To use this API, you just need to send `POST` HTTP messages to a talk session
endpoint. Each session is responsible for different personas as well as
contexts and conversation histories of the chatbot.

A session can be created just by accessing a random subpath of the server:
```
http://localhost:5003/some-random-talk-session
```

The server expects an UTF-8 json encoded payload containg the `text` key:
```
\"{
    \"text\":\"The message goes here\"
}\"
```

## Automatic setup
You can automatically configure the chatbot service by running `setup.sh` on the
root of the project.

## Manual setup
First create a virtual environment on the project root and activate it:
```
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:
```
pip install -e .
```

Initialize and setup the `transfer-learning-conv-ai` chatbot submodule:
```
git submodule init
cd transfer-learning-conv-ai
pip install -r requirements
python -m spacy download en
```

Create a directory for the resources (models and datasets)
```
mkdir -p resources/models
mkdir resources/datasets
```

Download the pretrained and fine-tuned model [here](https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz) and extract it inside the `models` directory:
```
cd resources/models
wget https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz
tar -xvf finetuned_chatbot_gpt.tar.gz
```

Download the Persona Chat dataset [here](https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json) and put it on `datasets` directory.
```
cd resources/datasets
wget https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz
```

If you created another directory for model or the dataset files, you must to change it `chatbot_service/src/config.json` properly.

## Stating the server
Finally, start the the server. By default, the service will run on port `5003`
```
./start-server
```
