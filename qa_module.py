from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class QuestionAnsweringBot:
    def __init__(self, model_name="deepset/bert-base-cased-squad2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)

    def answer_question(self, question, context):
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = inputs["input_ids"].tolist()[0]
        outputs = self.model(**inputs)

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )

        return answer.strip()
