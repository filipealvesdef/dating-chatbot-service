from setuptools import setup

setup(
    name='Dating chatbot API',
    description='An API that comunicates with a chatbot service',
    version='0.0.1',
    packages=[
        'chatbot_service',
    ],
    install_requires=[
        'flask~=1.1.2',
        'flask-cors~=3.0.8',
    ],
)
