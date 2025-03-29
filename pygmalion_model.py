from llama_cpp import Llama
import os

class PygmalionModel:
    def __init__(self):
        # Disable warnings
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        # Use Pygmalion 7B model instead of Mistral
        self.model_name = "TheBloke/Pygmalion-7B-GGUF"
        self.model_file = "pygmalion-2-7b.Q5_K_M.gguf"  # Ensure this matches your downloaded model

        # Path to the local model
        self.model_path = f"C:/AI/models/pygmalion-7b/{self.model_file}"

        # Load the model
        self.llm = Llama(
            model_path=self.model_path,  # Load local GGUF model
            n_ctx=2048,
            n_threads=6,  # Adjust for CPU efficiency
            n_gpu_layers=6 if os.getenv("USE_GPU") == "1" else 0  # Adjust GPU usage
        )

    def generate_response(self, prompt):
        """Generate a response based on the given prompt"""
        try:
            # Adjust the prompt format to fit Pygmalion’s style
            full_prompt = (
                f"### Instruction:\n"
                f"You are a helpful AI with the personality of a character named Eva.\n"
                f"Respond in a way that is {', '.join(['playful', 'sarcastic', 'empathetic'])}.\n\n"
                f"### User Input:\n{prompt}\n\n### Response:"
            )

            # Generate response
            output = self.llm(
                full_prompt,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                echo=False
            )

            return output["choices"][0]["text"].strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Oops, I couldn't generate a response!"

