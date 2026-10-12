# MiniMind — Independent AI Assistant

MiniMind is an experimental AI assistant developed in Python. Its core conversational model is a neural network trained from scratch using NumPy, without relying on pretrained language models or third-party AI inference services.

## Features

- Neural network trained from scratch for intent classification
- Customizable intents, patterns, and responses
- Persistent local memory
- Arithmetic calculator
- Offline search across local TXT and Markdown documents
- Local processing without a cloud AI API

## Technology Stack

- Python
- NumPy
- JSON
- Regular expressions
- Python AST and operator modules

## Project Status

MiniMind is under active development. Its current conversational model is an intent classifier, not a general-purpose large language model. Future development aims to improve language understanding, training methods, and conversational capabilities while keeping the model architecture and training process under my control.

## Installation

1. Install Python 3.
2. Clone this repository.
3. Open a terminal in the project directory.
4. Create and activate a virtual environment.
5. Install dependencies.
6. Train the model and run the chatbot.

Example commands for Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python chat.py
```

## Offline Operation

After Python, NumPy, and the project files are installed, the chatbot, calculator, memory, and document search can operate locally without a cloud AI service. The first-time installation of dependencies may require an internet connection.

## Contributions

Suggestions, bug reports, and contributions are welcome.

## License

The project's license will be specified in the LICENSE file.