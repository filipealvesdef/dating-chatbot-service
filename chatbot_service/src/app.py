from flask import Flask
from flask import request
from flask_cors import CORS

import json

app = Flask(__name__)
CORS(app)

@app.route('/<talk_session>', methods=['POST'])
def talk_session(talk_session):
    message = json.loads(request.data.decode('utf-8'))
    text = message['text']

    print(f'Talk session: {talk_session}')
    print(f'Message received: {text}')

    message['text'] += '\n\nHere we can preprocess the text, mapping acronyms,\
    as well as translating it, for example.'

    # TODO: Send the preprocessed text to the chatbot service and get its
    # response. Then, we must to retranslate the output text the original idiom.

    return json.dumps(message)
