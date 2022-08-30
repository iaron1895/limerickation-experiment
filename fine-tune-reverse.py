import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, random_split
import string as string_utils
from dataset import LimerickDataset

# code from https://github.com/coderalo/11785-automatic-poetry-generation/ repurposed for experiment
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

special_tokens = {
    "sep_token": "<LINE>",
    "pad_token": "<PAD>",
    "bos_token": "<BOS>"
}

tokenizer.add_special_tokens(special_tokens)

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

data = json.load(open("limericks.json"))
limericks = []

for _, limerick in data['limericks'].items():
    lines = limerick['lines']
    flag = True

    # Remove the final punctuation of each line
    # (we'll use a special separator instead)
    for idx, line in enumerate(lines):
        if len(line) == 0:
            flag = False
            break
        if line[-1] in string_utils.punctuation:
            lines[idx] = line[:-1]
    
    if flag:
        limericks.append(lines)

# create a limerick dataset that can be used by torch to fine tune gpt-2
dataset = LimerickDataset(limericks, tokenizer)

# dataset into train and validation
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_steps = 10000
)

trainer = Trainer(model=model,  args=training_args, train_dataset=train_dataset, 
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])})

# this can take up to a day to run
trainer.train()

model.save_pretrained('./Reversed_limerick_model')
tokenizer.save_pretrained('./Reversed_limerick_tokenizer/')
