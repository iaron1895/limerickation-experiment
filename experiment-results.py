from dataset import reverse_line
import torch
import copy
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModel, AutoModelForCausalLM

# code from https://github.com/coderalo/11785-automatic-poetry-generation/ repurposed for experiment

tokenizer = GPT2Tokenizer.from_pretrained('./Reversed_limerick_tokenizer/')
model = AutoModelForCausalLM.from_pretrained('./Reversed_limerick_model/')

def get_input_ids(prompt, tokenizer, use_bos, add_line_token):
    prompt = prompt.strip()
    if add_line_token:
        if prompt != "" and prompt[-6:] != "<LINE>":
            prompt += " <LINE>"
    if use_bos and prompt[:5] != "<BOS>":
        prompt = "<BOS> " + prompt

    input_ids = reverse_line(
        input_ids=tokenizer(prompt, return_tensors="np").input_ids[0],
        tokenizer=tokenizer,
        reverse_last_line=True)
    input_ids = torch.tensor(input_ids).reshape(1, -1)

    return input_ids


def batch_decode(outputs,tokenizer,reverse_last_line):
    reversed = []
    for output in outputs:
        output = torch.tensor(
            reverse_line(
                input_ids=output.cpu().numpy(),
                tokenizer=tokenizer,
                reverse_last_line=reverse_last_line)
            ).reshape(-1)
        reversed.append(output)
    outputs = torch.stack(reversed)

    outputs = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=False)

    return outputs

def count_lines(prompt):
    return len(prompt.strip().split("<LINE>")) - 1

def lengths_to_mask(lengths, dtype, device, position="pos"):
    max_len = lengths.max().item()
    if position == "pos":
        mask = torch.arange(
            max_len,
            dtype=lengths.dtype,
            device=lengths.device)
        mask = mask.expand(len(lengths), max_len)
        mask = (mask < lengths.unsqueeze(1))
    else:
        mask = torch.arange(
            max_len - 1, -1, -1,
            dtype=lengths.dtype,
            device=lengths.device)
        mask = mask.expand(len(lengths), max_len)
        mask = (mask < lengths.unsqueeze(1))

    mask = mask.clone().detach()
    mask = mask.to(dtype=dtype, device=device)
    
    return mask

def generate_lines(model, tokenizer, prompts, generate_params, num_generation, batch_size, add_line_token):
    use_bos = True
    full_input_ids = []
    num_lines = []
    for prompt in prompts:
        num_lines = count_lines(prompt)
        input_ids = get_input_ids(prompt=prompt,tokenizer=tokenizer,use_bos=use_bos,add_line_token=add_line_token)
        input_ids = input_ids.repeat(num_generation, 1)
        full_input_ids.append(input_ids)

    # generate attention mask
    lengths = []
    for input_ids in full_input_ids:
        lengths += [input_ids.shape[1]] * input_ids.shape[0]
    lengths = torch.tensor(lengths, dtype=torch.long)
    full_attention_mask = lengths_to_mask(lengths, torch.long, "cpu", "pre")

    # pad the input ids
    max_seq_len = max([input_ids.shape[1] for input_ids in full_input_ids])
    full_input_ids = [
        torch.cat([
            torch.full(
                (input_ids.shape[0], max_seq_len - input_ids.shape[1]),
                fill_value=tokenizer.eos_token_id, dtype=torch.long
            ),
            input_ids
        ], dim=1)
        for input_ids in full_input_ids]
    full_input_ids = torch.cat(full_input_ids, dim=0)

    num_batches = math.ceil(full_input_ids.shape[0] / batch_size)

    # assume that a line cannot be longer than 30 tokens
    tmp_params = copy.deepcopy(generate_params)
    if "max_length" in tmp_params:
        tmp_params.pop("max_length")
    tmp_params["max_new_tokens"] = 30

    # Step 2: pass the batch into model to get generation output
    outputs = []
    for i in range(num_batches):
        input_ids = full_input_ids[i * batch_size: (i + 1) * batch_size]
        input_ids = input_ids.to(device="cpu")
        attention_mask = \
            full_attention_mask[i * batch_size: (i + 1) * batch_size]
        attention_mask = attention_mask.to(device="cpu")
        with torch.no_grad():
            output = model.generate(input_ids, **tmp_params,attention_mask=attention_mask,pad_token_id=tokenizer.eos_token_id)
            output = torch.unbind(output)
            outputs.extend(output)
    
    # Step 3: convert the generation result back to strings
    outputs = batch_decode(outputs=outputs,tokenizer=tokenizer,reverse_last_line=False)

    print("Outputs are")
    print(outputs)
    clean_outputs = []
    for output in outputs:
        new_num_lines = count_lines(output)
        if new_num_lines < num_lines + 1:
            continue
        output = output.strip().split(" <LINE> ")[:num_lines + 1]
        output = " <LINE> ".join(output) + " <LINE>"
        output = output.replace("<|endoftext|>", "").strip()
        clean_outputs.append(output)
  
    return clean_outputs


def generate_new_lines(model,tokenizer,prompts,generate_params,num_generation,batch_size):
    return generate_lines(model=model,tokenizer=tokenizer,prompts=prompts,generate_params=generate_params,num_generation=num_generation,batch_size=batch_size,add_line_token=True)
    

def finish_lines(model,tokenizer,prompts,generate_params,num_generation,batch_size):
    return generate_lines(model=model,tokenizer=tokenizer,prompts=prompts,generate_params=generate_params,num_generation=num_generation,batch_size=batch_size,add_line_token=False)

def generate_limericks(model,tokenizer,prompts,generate_params,num_generation=10,batch_size=1,add_line_token=True):
    use_bos = True
    full_input_ids = []
    for prompt in prompts:
        input_ids = get_input_ids(prompt=prompt,tokenizer=tokenizer,use_bos=use_bos,add_line_token=add_line_token)
        input_ids = input_ids.repeat(num_generation, 1)
        full_input_ids.append(input_ids)

    # generate attention mask
    lengths = []
    for input_ids in full_input_ids:
        lengths += [input_ids.shape[1]] * input_ids.shape[0]
    lengths = torch.tensor(lengths, dtype=torch.long)
    full_attention_mask = lengths_to_mask(lengths, torch.long, "cpu", "pre")
    # pad the input ids
    max_seq_len = max([input_ids.shape[1] for input_ids in full_input_ids])
    full_input_ids = [
        torch.cat([
            torch.full(
                (input_ids.shape[0], max_seq_len - input_ids.shape[1]),
                fill_value=tokenizer.eos_token_id, dtype=torch.long
            ),
            input_ids
        ], dim=1)
        for input_ids in full_input_ids]
    full_input_ids = torch.cat(full_input_ids, dim=0)

    num_batches = math.ceil(full_input_ids.shape[0] / batch_size)

    # Step 2: pass the batch into model to get generation output
    outputs = []
    for i in range(num_batches):
    # for i in tqdm.trange(num_batches, leave=False):
        input_ids = full_input_ids[i * batch_size: (i + 1) * batch_size]
        input_ids = input_ids.to(device='cpu')
        attention_mask = \
            full_attention_mask[i * batch_size: (i + 1) * batch_size]
        attention_mask = attention_mask.to(device='cpu')
        with torch.no_grad():
            output = model.generate(
                input_ids, **generate_params,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id)
            output = torch.unbind(output)
            outputs.extend(output)

    # Step 3: convert the generation result back to strings
    outputs = batch_decode(outputs=outputs,tokenizer=tokenizer,reverse_last_line=False)
    clean_outputs = []
    for output in outputs:
        new_num_lines = count_lines(output)
        if new_num_lines < 5:
            continue
        output = output.strip().split(" <LINE> ")[:5]
        output = " <LINE> ".join(output) + " <LINE>"
        # clean up the prepended tokens
        output = output.replace("<|endoftext|>", "").strip()
        clean_outputs.append(output)

    return clean_outputs


# code to return limericks generated from the first verse using reverse model
generate_params = {
    "do_sample": True,
    "max_length": 100,
}

results = []
for _ in range(10):
    results.append(
        generate_limericks(
            model,
            tokenizer,
            ["There was a positive boy named Pete"],
            generate_params,
            num_generation=1,
            batch_size=1,
            add_line_token=True)[0])

for res in results:
    print(res)

# code to return limericks generated from the first verse using forward model

model = GPT2LMHeadModel.from_pretrained('./Fine_tune_limerick_model')
tokenizer = GPT2Tokenizer.from_pretrained('./Fine_tune_limerick_tokenizer/')

generated = tokenizer("<|startoftext|> ", return_tensors="pt").input_ids


inputs = tokenizer("<|startoftext|> There was a positive boy named", return_tensors="pt").input_ids
outputs = model.generate(inputs, max_length = 300, top_p = 0.95, temperature = 1.9, do_sample = True, num_return_sequences = 10,top_k=50)

for i, sample_output in enumerate(outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))