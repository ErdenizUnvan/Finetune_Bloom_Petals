import torch
import transformers
from transformers import BloomTokenizerFast
from transformers import BloomForTokenClassification
from transformers import BloomForCausalLM

model = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b1")
tokenizer = BloomForCausalLM.from_pretrained("bigscience/bloom-1b1")

prompt="What does the future hold for my professional career as a Data Scientist?"
result_length=150
inputs=tokenizer(prompt, return_tensors="pt")
print(inputs)
