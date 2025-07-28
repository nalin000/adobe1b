from transformers import pipeline, set_seed

class Generator:
    def __init__(self, model_name):
        self.generator = pipeline("text-generation", model=model_name, device=-1)
        set_seed(42)

    def generate_summary(self, context: str, question: str = "") -> str:
        # Skip garbage chunks
        if not context or len(context.split()) < 30:
            return ""  # Ignore very short or empty chunks

        prompt = (
            "You are a professional travel planner helping organize a 4-day group trip for 10 college friends. "
            "From the context below, extract any useful suggestions — activities, places to visit, food experiences, nightlife, "
            "cultural tips, or travel advice. Be concise and useful for planning. Avoid repetition.\n\n"
            f"Context:\n{context}\n\n"
            "Extracted Points:"
        )

        output = self.generator(prompt, max_length=200, do_sample=True, temperature=0.7)
        response = output[0]["generated_text"]

        # Extract the part after "Extracted Points:"
        parts = response.split("Extracted Points:")
        answer = parts[-1].strip() if len(parts) > 1 else response.strip()

        # Filter out junk like repetition or nonsense
        if "no, no" in answer.lower() or len(set(answer.split())) / (len(answer.split()) + 1e-5) < 0.5:
            return ""  # Discard likely bad output

        return answer
