"""
LLM client supporting OpenAI API and Ollama-compatible local models.

Set environment variables:
    LLM_PROVIDER  = "openai" or "ollama"  (default: "openai")
    OPENAI_API_KEY = your OpenAI key       (required if provider is openai)
    OPENAI_MODEL   = model name            (default: "gpt-4o-mini")
    OLLAMA_BASE_URL = Ollama server URL    (default: "http://localhost:11434")
    OLLAMA_MODEL    = model name           (default: "llama3")
"""

import json
import os

import requests
from openai import OpenAI


def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower()


def _call_openai(prompt: str, system_prompt: str) -> str:
    """Call the OpenAI Chat Completions API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str, system_prompt: str) -> str:
    """Call an Ollama-compatible local model."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")

    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 4096,
        },
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def generate(prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to the configured LLM and return the response text.

    Args:
        prompt: The user-facing prompt string.
        system_prompt: Optional system-level instructions.

    Returns:
        The LLM's response as a string.
    """
    provider = _get_provider()

    if provider == "openai":
        return _call_openai(prompt, system_prompt)
    elif provider == "ollama":
        return _call_ollama(prompt, system_prompt)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'openai' or 'ollama'."
        )


def parse_json_response(text: str) -> dict:
    """
    Attempt to extract and parse JSON from an LLM response.

    The response may include markdown code fences or extra text around the JSON.
    """
    # Try to find JSON block in markdown fences
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Return raw text wrapped in a dict
    return {"raw_response": text}
