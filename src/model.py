"""
Model class for vLLM inference.
"""

import modal
import traceback
from .config import AVAILABLE_MODELS, DEFAULT_MODEL_ID


def create_model_class(model_id: str):
    """
    Factory function to create a Model class for a specific model_id.
    Each model gets its own Modal class with its own container.
    """
    model_config = AVAILABLE_MODELS.get(model_id)
    if not model_config:
        raise ValueError(f"Model {model_id} not found in AVAILABLE_MODELS")
    
    class Model:
        """vLLM model server that scales to zero."""
        
        # Store model_id as class variable (closure)
        _model_id = model_id
        _model_config = model_config

        @modal.enter()
        def load_model(self):
            """Load model once when container starts (cached across requests)."""
            from vllm import LLM, SamplingParams

            print(f"Loading model: {self._model_id}")
            print(f"Config: GPU={self._model_config.gpu_type}, MaxLen={self._model_config.max_model_len}")
            
            self.llm = LLM(
                model=self._model_id,
                max_model_len=self._model_config.max_model_len,
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
            import logging
            
            logger = logging.getLogger("llm-api")
            
            try:
                # Validate input
                if not messages or not isinstance(messages, list):
                    raise ValueError("messages must be a non-empty list")
                
                # Format messages for the model
                prompt = self._format_messages(messages)
                logger.debug(f"chat_stream: Formatted prompt length: {len(prompt)}")
                
                if not prompt or not prompt.strip():
                    raise ValueError("Formatted prompt is empty")
                
                sampling = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # vLLM's LLM.generate() returns a list, not a generator
                # We need to use the request_id to track streaming
                # For now, we'll generate and simulate streaming by chunking the output
                logger.debug("chat_stream: Starting vLLM generation")
                
                # Use generate which returns RequestOutput objects
                # The outputs list contains the final result
                request_outputs = self.llm.generate([prompt], sampling, use_tqdm=False)
                
                if not request_outputs:
                    raise ValueError("vLLM returned empty outputs")
                
                request_output = request_outputs[0]
                if not hasattr(request_output, 'outputs') or not request_output.outputs:
                    raise ValueError("RequestOutput has no outputs")
                
                # Get the final generated text
                final_text = request_output.outputs[0].text
                
                # Simulate streaming by chunking the text into smaller pieces
                # This is a workaround since vLLM's synchronous LLM doesn't support true streaming
                # For true streaming, we'd need to use AsyncLLMEngine (which requires async methods)
                chunk_size = 10  # Characters per chunk
                chunks = []
                for i in range(0, len(final_text), chunk_size):
                    chunk = final_text[i:i + chunk_size]
                    if chunk:
                        chunks.append(chunk)
                
                # If text is very short, return it as a single chunk
                if not chunks and final_text:
                    chunks = [final_text]
                
                logger.debug(f"chat_stream: Generated {len(chunks)} chunks from text of length {len(final_text)}")
                
                return chunks
                
            except Exception as e:
                logger.error(f"chat_stream: Error in chat_stream method: {str(e)}")
                logger.error(f"chat_stream: Traceback: {traceback.format_exc()}")
                raise

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
