import torch
from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM

MODEL_NAME = "bigscience/bloom-7b1-petals"
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
model = model.cuda()

# Save the model and tokenizer to your working directory
model.save_pretrained("my-bloom7b1-petals")
tokenizer.save_pretrained("my-bloom7b1-petals")
