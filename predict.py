"""
Generate text from GPT language model
"""
import torch
from transformers import GPT2TokenizerFast
from GPT import GPT


def generate_text(model, context=" ", top_p=0.9, temperature=1, max_len=100, device='cpu'):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    seq_len = model.seq_len

    show(context, color=False)
    # inference
    model.eval()
    for i in range(max_len):
        # (seq_len)
        # truncate context to max seq_len
        tokens = tokenizer.tokenize(context)
        tokens = tokens[-seq_len:]              
        context = tokenizer.convert_tokens_to_string(tokens)
        # encode text
        context_tokens = tokenizer(context, 
                                    padding='max_length',
                                    max_length=seq_len,
                                    return_tensors='pt'
                                    ).to(device)
        input = context_tokens['input_ids']
        attn_mask = context_tokens['attention_mask']
        # next token prediction
        logits, attn = model(input)                             # logit: (1, seq_len, vocab)
        next_token_idx = attn_mask.to(torch.int32).sum() - 1  
        logits = logits.squeeze()                               # (seq_len, vocab)
        token_logits = logits[next_token_idx]                   # (vocab)
        # sample token
        next_token = nucleus_sampling(token_logits, top_p, temperature)
        pred_word = tokenizer.decode(next_token)
        show(pred_word, color=True)
        # auto-regressive generation
        context += pred_word


def nucleus_sampling(logits, top_p, temperature):
    # logits: (vocab)
    temp_logits = logits / temperature
    probs = torch.softmax(temp_logits, dim=-1)  # (vocab)
    token = top_p_sampling(probs, top_p)
    return token


def top_p_sampling(probabilities, top_p):
    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    # Compute the cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Find the indices to keep
    top_p = max(top_p, cumulative_probs[0].item())
    keep_indices = (cumulative_probs <= top_p).nonzero().view(-1)
    # Select the top-p probabilities and indices
    top_p_probs = sorted_probs[keep_indices].view(-1)
    top_p_indices = sorted_indices[keep_indices].view(-1)
    # Normalize the probabilities
    normalized_probs = top_p_probs / top_p_probs.sum(dim=-1, keepdim=True).view(-1)
    # Sample from the normalized probabilities
    p_sample_idx = torch.multinomial(normalized_probs, num_samples=1, replacement=True)
    # get corresponding token id
    token = top_p_indices[p_sample_idx]
    return token



def greedy_sampling(logits):
    # greedy sampling -> leads to repeatition
    pred_token = torch.argmax(logits, dim=-1)    # (seq_len) 
    return pred_token


def show(text, color=False):
    # TODO - bug, text is not printed unless newline is printed
    c = e = ''
    if color:
        c = '\033[36m'
        e = '\033[0m'
    print(c + text + e, end='')
    

if __name__ == '__main__':
    context = """import num"""
    gpt = torch.load('model/tinygpt_codeparrot.pt')
    generate_text(gpt, context, 
                  top_p=0.9,
                  temperature=1, 
                  max_len=128,
                  device='mps')
