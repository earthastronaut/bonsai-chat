"""
Model class for vLLM inference.
"""

import modal
from .config import MODEL_ID, GPU_TYPE, MAX_MODEL_LEN, CONTAINER_IDLE_TIMEOUT, MAX_CONTAINERS
from .images import vllm_image


def create_model_class(app: modal.App):
    """Create and return the Model class with Modal decorators."""
    
    @app.cls(
        gpu=GPU_TYPE,
        image=vllm_image,
        scaledown_window=CONTAINER_IDLE_TIMEOUT,
        # secrets=[modal.Secret.from_name("huggingface")],  # Optional: only needed for private/gated models
        max_containers=MAX_CONTAINERS,
    )
    class Model:
        """vLLM model server that scales to zero."""

        @modal.enter()
        def load_model(self):
            """Load model once when container starts (cached across requests)."""
            from vllm import LLM, SamplingParams

            print(f"Loading model: {MODEL_ID}")
            self.llm = LLM(
                model=MODEL_ID,
                max_model_len=MAX_MODEL_LEN,
                trust_remote_code=True,
            )
            self.default_sampling = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            )
            print("Model loaded!")

        @modal.method()
        def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7):
            """Generate completion for a prompt."""
            from vllm import SamplingParams

            sampling = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            outputs = self.llm.generate([prompt], sampling)
            return outputs[0].outputs[0].text

        @modal.method()
        def chat(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7):
            """Chat completion (OpenAI format)."""
            from vllm import SamplingParams

            # Format messages for the model
            prompt = self._format_messages(messages)
            
            sampling = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            outputs = self.llm.generate([prompt], sampling)
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": outputs[0].outputs[0].text
                    }
                }]
            }

        @modal.method()
        def chat_stream(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7):
            """Stream chat completion (OpenAI format). Returns list of text deltas."""
            from vllm import SamplingParams

            # Format messages for the model
            prompt = self._format_messages(messages)
            
            sampling = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Use streaming generation - vLLM yields RequestOutput objects incrementally
            stream = self.llm.generate([prompt], sampling, use_tqdm=False)
            
            # Collect text deltas as they're generated
            chunks = []
            previous_text = ""
            for output in stream:
                # Each output contains the full text so far, extract the delta
                for choice in output.outputs:
                    current_text = choice.text
                    # Calculate delta (new text since last chunk)
                    if len(current_text) > len(previous_text):
                        delta = current_text[len(previous_text):]
                        if delta:
                            chunks.append(delta)
                            previous_text = current_text
            
            return chunks

        def _format_messages(self, messages: list[dict]) -> str:
            """Format chat messages for Mistral-style models."""
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"[INST] {content} [/INST]\n"
                elif role == "user":
                    formatted += f"[INST] {content} [/INST]\n"
                elif role == "assistant":
                    formatted += f"{content}\n"
            return formatted
    
    return Model

