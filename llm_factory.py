
import os
import time
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()


MODEL_OPTIONS = [
    "OpenAI (GPT-4o-mini)",
    "Groq: GPT-OSS 120B",
    "Groq: Qwen 27B",
    "Google Gemini",
    "Hugging Face (Mistral NeMo 12B)"
]


def get_llm(model_choice: str = "OpenAI (GPT-4o-mini)", temperature: float = 0):
    """
    Returns the ChatModel matching the user's selected provider.
    """

    # 1. Groq - Qwen 27B
    if "Qwen" in model_choice:
        return ChatGroq(
            model_name="qwen/qwen3.8-27b",
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    # 2. Groq - GPT-OSS 120B
    elif "Groq" in model_choice or "GPT-OSS" in model_choice:
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(
            model_name=model,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    # 3. Google Gemini
    elif "Gemini" in model_choice:
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from .env")

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    # 4. Hugging Face - Mistral NeMo 12B
    elif "Hugging Face" in model_choice:
        model = os.getenv(
            "MODEL_NAME",
            "nvidia/Mistral-NeMo-12B-Instruct"
        )

        return ChatOpenAI(
            base_url="https://router.huggingface.co/hf-inference/v1",
            api_key=os.getenv("HF_TOKEN"),
            model=model,
            temperature=temperature
        )

    # 5. Default OpenAI - GPT-4o-mini
    else:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )


def extract_metrics(response: Any, start_time: float) -> Dict[str, Any]:
    """
    Extracts response latency (seconds) and token counts.
    """

    latency = round(time.time() - start_time, 2)

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = response.usage_metadata.get("input_tokens", 0)
        output_tokens = response.usage_metadata.get("output_tokens", 0)
        total_tokens = response.usage_metadata.get("total_tokens", 0)

    elif hasattr(response, "response_metadata") and response.response_metadata:
        token_usage = response.response_metadata.get("token_usage", {})

        input_tokens = (
            token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or 0
        )

        output_tokens = (
            token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0
        )

        total_tokens = (
            token_usage.get("total_tokens")
            or (input_tokens + output_tokens)
        )

    tokens_per_sec = (
        round(output_tokens / latency, 1)
        if latency > 0 and output_tokens > 0
        else 0
    )

    return {
        "latency_sec": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec
    }




