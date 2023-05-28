import os
import torch
import transformers
import wandb
from datasets import load_dataset
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BloomTokenizerFast, get_scheduler

from petals import DistributedBloomForCausalLM

print('Bütün kütüphaneler import edildi')
# Choose a model you'd like to prompt-tune. We recommend starting with
# the smaller 7.1B version of BLOOM (bigscience/bloom-7b1-petals) for faster prototyping.
# Once your code is ready, you can switch to full-scale
# 176B-parameter BLOOM (bigscience/bloom-petals) or BLOOMZ (bigscience/bloomz-petals).
print('Bloom 7B modeli yükleniliyor.')

MODEL_NAME = "bigscience/bloom-7b1-petals"
# Choose a prompt-tuning mode ('ptune' or 'deep_ptune').
# The latter fine-tunes separate prefixes for each transformer block,
# so prompt-tuning will take more time but yield better results.
# See this paper for details of how it works: https://arxiv.org/pdf/2110.07602.pdf
TUNING_MODE = 'ptune'

NUM_PREFIX_TOKENS = 16
DEVICE = 'cuda'
BATCH_SIZE = 8
LR = 1e-2
WEIGHT_DECAY = 0.0
NUM_SAMPLES = 1000
SEED = 42
MODEL_MAX_LENGTH = 256
print('Bloom 7B için model parametreleri hazır.')

tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'right'
tokenizer.model_max_length = MODEL_MAX_LENGTH
model = DistributedBloomForCausalLM.from_pretrained(
    MODEL_NAME,
    pre_seq_len=NUM_PREFIX_TOKENS,
    tuning_mode=TUNING_MODE
).to(DEVICE)
print('Bloom 7B modeli yüklendi.')

print('Dataset yükleniliyor.')
dataset = load_dataset("bavard/personachat_truecased")
print('Dataset yüklendi.')

print('Sohbet Botu için fonksiyonlar hazırlanılıyor.')
def chunking(examples):
    inputs = [
        "\n-----\n".join(history) + "\n-----\n" + candidate
        for history, candidates in zip(examples["history"], examples["candidates"])
        for candidate in candidates
    ]
    return {"chunks": inputs}


def tokenize(examples):
    outputs = {
        "input_ids": tokenizer(examples["chunks"], padding='max_length', truncation=True)["input_ids"]
    }
    outputs["labels"] = outputs["input_ids"]
    return outputs
print('Sohbet Botu için fonksiyonlar hazır.')

print('Dataseti ile finetuning yapılması için model hazırlanılıyor.')

tokenized_datasets = (
    dataset
        .map(chunking, batched=True, remove_columns=dataset["train"].column_names)
        .map(tokenize, batched=True, remove_columns=["chunks"])
)

tokenized_datasets.set_format("torch")
train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
train_dataloader = DataLoader(
    train_dataset.select(list(range(NUM_SAMPLES))),
    shuffle=True,
    batch_size=BATCH_SIZE,
    drop_last=True,
)
print('Dataseti ile finetuning yapılması için model hazırlandı.')

for n, p in model.named_parameters():
    if p.requires_grad:
        print(n, p.requires_grad, p.device)

print('Modelin eğitilecek parametrelerinin kontrolleri yapıldı.')

print('Optimizer ve learning rate scheduler hazırlanılıyor.')

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)
)

print('Optimizer ve learning rate scheduler hazır.')

print('wandb başlıyor.')


wandb.init(
    project="bloom-personachat",
    config={
        "num_samples": NUM_SAMPLES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_prefix_tokens": NUM_PREFIX_TOKENS,
        "model_name": MODEL_NAME,
        "seed": SEED,
    }
)

print("Wandb'de eğitim ve loglama başlıyor.")

for batch in tqdm(train_dataloader):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    wandb.log({"Train Loss": loss})
#torch.save(model.state_dict(), 'model.bin')
wandb.save('model.bin')
#model.cuda()

# Save the model and tokenizer to your working directory
#model.save_pretrained("my_ft_bloom_model")
#tokenizer.save_pretrained("my_ft_bloom_model")


print("Wandb'de eğitim ve loglama tamamlandı.")

print('Sohbet Botu hazır: ')

TOP_K = 100
TEMPERATURE = 0.6

with model.inference_session(max_length=512) as sess:
    while True:
        user_phrase = input()
        if len(user_phrase) == 0:
            break
        inputs = tokenizer([f"{user_phrase}\n-----\n"], return_tensors='pt')['input_ids'].to(DEVICE)
        while True:
            outputs = model.generate(
                inputs,
                temperature=TEMPERATURE,
                do_sample=True,
                top_k=TOP_K,
                max_new_tokens=1,
                session=sess,
            )
            bloom_answer_token = tokenizer.decode(outputs[0, -1:])
            print(bloom_answer_token, end="", flush=True)
            if bloom_answer_token == "\n":
                break
            inputs = None
