"""
Generate text from GPT language model
"""
import torch
from transformers import GPT2TokenizerFast
from GPT import GPT

# TODO - slidding window text generation
# TODO impelemnt nucleus decoding - top_p
def generate_text(model, context=" ", max_len=100, device='cpu'):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    show(context, color=False)
    # inference
    model.eval()
    for i in range(max_len):
        # (seq_len)
        context_tokens = tokenizer(context, 
                                    padding='max_length',
                                    max_length=model.seq_len,
                                    return_tensors='pt'
                                    ).to(device)
        input = context_tokens['input_ids']
        attn_mask = context_tokens['attention_mask']
        # next token prediction
        logits, attn = model(input)                             # logit: (1, seq_len, vocab)
        next_token_idx = attn_mask.to(torch.int32).sum() - 1  
        logits = logits.squeeze()                               # (seq_len, vocab)
        token_logits = logits[next_token_idx]                # (vocab)
        # sample token
        next_token = nucleus_sampling(token_logits, top_p=0.9, temperature=0.3)
        pred_word = tokenizer.decode(next_token)
        show(pred_word, color=True)
        # auto-regressive generation
        context += pred_word

        # TODO - allow moving window
        if next_token_idx == model.seq_len-1:
            break


def nucleus_sampling(logits, top_p=0.9, temperature=1):
    # logits: (vocab)
    temp_logits = logits / temperature
    probs = torch.softmax(temp_logits, dim=-1)  # (vocab)
    token = top_p_sampling(probs, top_p)
    return token


def top_p_sampling(probabilities, top_p=0.9):
    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    # Compute the cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Find the indices to keep
    keep_indices = (cumulative_probs <= top_p).nonzero()
    # Select the top-p probabilities and indices
    top_p_probs = sorted_probs[keep_indices].squeeze()
    top_p_indices = sorted_indices[keep_indices].squeeze()
    # Create a mask to zero-out all other probabilities
    mask = torch.zeros_like(probabilities).scatter(1, top_p_indices.unsqueeze(1), 1.0)
    # Apply the mask to the probabilities
    masked_probs = mask * probabilities
    # Normalize the probabilities
    normalized_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
    # Sample from the normalized probabilities
    samples = torch.multinomial(normalized_probs, num_samples=1)
    return samples.squeeze()



def greedy_sampling(logits):
    # greedy sampling -> leads to repeatition
    pred_token = torch.argmax(logits, dim=-1)    # (seq_len) 
    return pred_token


def show(text, color=False):
    c = e = ''
    if color:
        c = '\033[36m'
        e = '\033[0m'
    print(c + text + e, end='')
    

if __name__ == '__main__':
    context = """God is"""
    gpt = torch.load('model/tinygpt.pt')
    generate_text(gpt, context, device='mps')
