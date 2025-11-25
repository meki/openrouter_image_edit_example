# OpenRouter Gemini Pro 3 Image Editing Example

This project demonstrates how to use OpenRouter's Gemini Pro 3 model for image editing tasks. It includes an example script that takes an input image and a text prompt to modify the image accordingly.

## Setup

uv Create a virtual environment and install dependencies:

```bash
uv venv
uv sync
```

```bash
cp .env.example .env
# Edit the .env file to add your OpenRouter API key and desired output folder path.
```

## Usage

Copy your input image to the specified path in `prompt_info.yaml` and modify the text prompt as needed. Then run the main script:

```bash
uv run python main.py
```