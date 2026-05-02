"""
Unified LLM interface for multiple models.
Supports: Gemini, ChatGPT (OpenAI)
"""

from abc import ABC, abstractmethod
import os


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response text
        """
        pass


class GeminiModel(LLMInterface):
    """Gemini model via Google API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini model.
        
        Args:
            api_key: Google API key (loads from env if not provided)
        """
        import google.generativeai as genai
        
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-flash-latest")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate response using Gemini.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)

        Returns:
            Generated response
        """
        import time
        import re as _re
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 500,
                    }
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    match = _re.search(r'seconds: (\d+)', error_str)
                    wait_time = int(match.group(1)) + 5 if match else 65
                    print(f"      Rate limited. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries - 1})...")
                    time.sleep(wait_time)
                else:
                    print(f"Error generating response: {e}")
                    return "Error generating response"
        return "Error generating response"


# class BioMedLMModel(LLMInterface):
#     """BioMedLM model from Hugging Face."""
#
#     def __init__(self):
#         from transformers import AutoTokenizer, AutoModelForCausalLM
#         from huggingface_hub import login
#         model_name = "stanford-crfm/BioMedLM"
#         hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
#         if hf_token:
#             try:
#                 login(token=hf_token, add_to_git_credential=False)
#             except Exception:
#                 pass
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=hf_token)
#
#     def generate(self, prompt: str, max_length: int = 200) -> str:
#         try:
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#             outputs = self.model.generate(**inputs, max_length=max_length, temperature=0.7,
#                                           top_p=0.95, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
#             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             if prompt in response:
#                 response = response.replace(prompt, "").strip()
#             return response
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "Error generating response"


# class LLaMAModel(LLMInterface):
#     """LLaMA 3.1 model from Hugging Face."""
#
#     def __init__(self):
#         from transformers import pipeline
#         from huggingface_hub import login
#         self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
#         self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
#         if self.hf_token:
#             try:
#                 login(token=self.hf_token, add_to_git_credential=False)
#             except Exception:
#                 pass
#         self.pipeline = pipeline("text-generation", model=self.model_name, device_map="auto", token=self.hf_token)
#
#     def generate(self, prompt: str, max_length: int = 200) -> str:
#         try:
#             messages = [{"role": "user", "content": prompt}]
#             response = self.pipeline(messages, max_length=max_length, temperature=0.7, top_p=0.95, do_sample=True)
#             generated = response[0]["generated_text"]
#             if isinstance(generated, list) and generated:
#                 generated = generated[-1].get("content", "")
#             if isinstance(generated, str) and prompt in generated:
#                 generated = generated.replace(prompt, "").strip()
#             return generated if isinstance(generated, str) else str(generated)
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "Error generating response"


class ChatGPTModel(LLMInterface):
    """ChatGPT model via OpenAI API."""

    def __init__(self, api_key: str = None):
        """
        Initialize ChatGPT model.

        Args:
            api_key: OpenAI API key (loads from env if not provided)
        """
        from openai import OpenAI

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate response using ChatGPT (gpt-4o-mini).

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)

        Returns:
            Generated response
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"


class ModelFactory:
    """Factory for creating LLM instances."""
    
    _models = {
        "gemini": GeminiModel,
        "chatgpt": ChatGPTModel,
        # "biomedlm": BioMedLMModel,
        # "llama": LLaMAModel,
    }

    @classmethod
    def create_model(cls, model_name: str) -> LLMInterface:
        """
        Create an LLM instance.

        Args:
            model_name: Name of the model ('gemini', 'chatgpt')
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If model not supported
        """
        model_name = model_name.lower()
        
        if model_name not in cls._models:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Supported models: {list(cls._models.keys())}"
            )
        
        model_class = cls._models[model_name]
        return model_class()


def query_model(model_name: str, prompt: str, temperature: float = 0.7) -> str:
    """
    Unified interface for querying any model.
    
    Args:
        model_name: Name of the model
        prompt: Input prompt
        temperature: Temperature for sampling
        
    Returns:
        Generated response
    """
    model = ModelFactory.create_model(model_name)
    
    if model_name.lower() in ("gemini", "chatgpt"):
        return model.generate(prompt, temperature=temperature)
    else:
        return model.generate(prompt)
