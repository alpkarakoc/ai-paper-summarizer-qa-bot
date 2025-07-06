from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class PaperSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def summarize(self, text, max_length=250, min_length=50):
        inputs = self.tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(self.device)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
