# Run Dynasor locally with ollama + OpenWebUI

This tutorial shows how to deploy Dynasor locally with ollama and deepseek-r1.

## Install ollama

Install [ollama](https://ollama.com/download) and pull a [deepseek-r1](https://ollama.com/library/deepseek-r1) model.
```bash
ollama run deepseek-r1
```

## Install Dynasor

```bash
git clone https://github.com/hao-ai-lab/Dynasor.git
cd Dynasor
pip install .
```

## Run `dynasor-openai`

We also provide a proxy server `dynasor-openai` for Dynasor that is compatible with any OpenAI API.

```bash
dynasor-openai --base-url http://localhost:11434/v1 --model deepseek-r1 --port 8001
```

You can interact with the proxy server running `dynasor-chat` with the base URL `http://localhost:8001/v1`.

```bash
dynasor-chat --base-url http://localhost:8001/v1
```

or simply run one of our example scripts to verify the proxy server is working:

```bash
python examples/client.py --prompt "2+2=?" --base-url http://localhost:8001/v1
```

## Use Dynasor with OpenWebUI

Install [Open WebUI](https://github.com/open-webui/open-webui) and run the server
```bash
pip install open-webui
open-webui serve
```

Then follow this instruction to [add a custom API](https://docs.openwebui.com/tutorials/integrations/amazon-bedrock#step-3-add-connection-in-open-webui) to Open WebUI.
- URL: `http://localhost:8001/v1` (the base URL of the proxy server)
- Key: `EMPTY` (optional)
- Prefix ID: `dynasor` (optional)


