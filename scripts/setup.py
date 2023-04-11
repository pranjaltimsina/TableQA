from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch

model_name = "google/tapas-base-finetuned-sqa"
model = TapasForQuestionAnswering.from_pretrained(model_name)
output_model_file = '../models/tapas.bin'
output_vocab_file = '../models/tapas_vocab.bin'
model_name = "google/tapas-base-finetuned-sqa"
tokenizer = TapasTokenizer.from_pretrained(model_name)
torch.save(model, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)
