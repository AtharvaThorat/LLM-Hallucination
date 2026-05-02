import os
import time
from dotenv import load_dotenv

load_dotenv()

_llama_pipeline = None


def _get_llama_pipeline():
    global _llama_pipeline
    if _llama_pipeline is None:
        from transformers import pipeline
        # Qwen2.5-0.5B-Instruct is not gated — no HF_TOKEN required
        # torch_dtype omitted: transformers picks the best dtype automatically
        _llama_pipeline = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
        )
    return _llama_pipeline


def call_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    block = next(b for b in message.content if isinstance(b, anthropic.types.TextBlock))
    return block.text.strip()


def call_gpt(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def call_gemini(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return (response.text or "").strip()


def call_llama(prompt: str) -> str:
    pipe = _get_llama_pipeline()
    messages = [{"role": "user", "content": prompt}]
    result = pipe(messages, max_new_tokens=256)
    # Pipeline returns full conversation — last message is always the assistant reply
    generated = result[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    return str(generated).strip()


def query_model(model_name: str, prompt: str, retries: int = 2) -> str:
    dispatchers = {
        "claude": call_claude,
        "gpt": call_gpt,
        "gemini": call_gemini,
        "llama": call_llama,
    }
    fn = dispatchers.get(model_name.lower())
    if fn is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(dispatchers.keys())}")

    last_error = ""
    for attempt in range(retries + 1):
        try:
            return fn(prompt)
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return f"[ERROR: {model_name} failed after {retries + 1} attempts — {last_error}]"
