# TGPT
TG LLM chat bot

## Installation and Usage
1. Clone the repository and navigate into it
2. Create a Python virtual environment by running in the terminal `python -m venv .venv` and activate it
3. Install the Python requirements by running `pip install -r requirements.txt`. Install ollama as well (https://ollama.com/)
4. Rename `exampleconfig.json` to `config.json`. Change its contents to your needs
5. Run `main.py`

Linux:
```sh
git clone https://github.com/mxurlx/tgpt.git
cd tgpt
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
mv exampleconfig.json config.json
# Don't forget to edit config.json
./.venv/bin/python main.py
```

## Uninstallation
Remove the cloned tgpt directory. You can also remove ollama.
