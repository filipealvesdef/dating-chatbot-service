from flask import Flask
from flask import request
from flask_cors import CORS

import json
from chitchat import send_message, load_model_config

app = Flask(__name__)
CORS(app)
model_config = load_model_config()
histories = {}


@app.route('/<talk_session>', methods=['POST'])
def talk_session(talk_session):
    message = json.loads(request.data.decode('utf-8'))
    text = message['text']

    if talk_session not in histories:
        histories[talk_session] = {
            'messages': [],
            'personality': None,
        }

    print(f'Talk session: {talk_session}')
    print(f'Message received: {text}')
    print(histories)

    # TODO: Preprocess the text to the chatbot service. (Translate, remove
    # acronyms, etc

    response_msg = send_message(histories[talk_session], text, model_config)
    print(response_msg)

    # TODO: Translate the text back to the original idiom

    return json.dumps({
        'text': response_msg,
    })
