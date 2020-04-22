python -m venv venv
source venv/bin/activate
pip install -e .
git submodule init
git submodule update
cd transfer-learning-conv-ai
pip install -r requirements.txt
python -m spacy download en
cd ..
mkdir -p resources/models
mkdir -p resources/datasets
cd resources/models
wget https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz
tar -xvf gpt_personachat_cache.tar.gz
cd ../datasets; wget https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json
cd ..
