from llama_cpp import Llama
import os

class NeuroSamaModel:
    def __init__(self):
        # Disable warnings
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        # Use a compatible model (Mistral as example)
        self.model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        self.model_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        
        # Download and load the model
        self.llm = Llama.from_pretrained(
            repo_id=self.model_name,
            filename=self.model_file,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=40 if os.getenv("USE_GPU") == "1" else 0
        )

    def generate_response(self, prompt):
        """Generate response from user input"""
        try:
            # Format prompt for instruction-following
            full_prompt = f"[INST] {prompt} [/INST]"
            
            # Generate response
            output = self.llm(
                full_prompt,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                echo=False
            )
            
            return output['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            return None