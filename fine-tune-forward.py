import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, random_split

def filter_limericks():
    f = open('limerick_dataset_oedilf_v3.json', "r")
    data  = json.loads(f.read())
    output_dict = [x for x in data if x['is_limerick']]

    with open('limerick_dataset.json', 'w') as f:
        json.dump(output_dict, f, indent = 2)

# uncomment below line to filter the limericks dataset
# filter_limericks()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

f = open('limerick_dataset.json')
data = json.load(f)
limericks = [x['limerick'] for x in data]

max_length = max([len(tokenizer.encode(limerick)) for limerick in limericks])

class LimerickDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# create a limerick dataset that can be used by torch to fine tune gpt-2
dataset = LimerickDataset(limericks, tokenizer, max_length=max_length)

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

model.save_pretrained('./Fine_tune_limerick_model')
tokenizer.save_pretrained('./Fine_tune_limerick_tokenizer/')
