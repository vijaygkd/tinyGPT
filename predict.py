"""
Generate text from GPT language model
"""
import torch
from transformers import GPT2TokenizerFast
from dataset import CharTokenizer


def generate_text(model, tokenizer, context=" ", top_p=0.9, temperature=1, max_len=100, device='cpu'):
    seq_len = model.seq_len

    show(context, color=False)
    # inference -> turn off dropout
    model.eval()
    for i in range(max_len):
        # (seq_len)
        # truncate context to max seq_len
        context = context[-seq_len:]
        output_token_idx = len(context)-1
        # encode text
        context_tokens = tokenizer.encode(context)
        # pad to seq_len
        context_tokens = context_tokens + [tokenizer.eos_token_id] * (seq_len - len(context_tokens))
        context_tokens = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).to(device)
        # get input and attention mask
        input = context_tokens
        
        logits, attn = model(input)                             # logit: (1, seq_len, vocab)
        logits = logits.squeeze()                               # (seq_len, vocab)
        token_logits = logits[output_token_idx]                   # (vocab)
        
        # sample token
        next_token = nucleus_sampling(token_logits, top_p, temperature)
        # next_token = greedy_sampling(token_logits)

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
    return torch.tensor([pred_token])


def show(text, color=False):
    # TODO - bug, text is not printed unless newline is printed
    c = e = ''
    if color:
        c = '\033[36m'
        e = '\033[0m'
    print(c + text + e, end='')
    

if __name__ == '__main__':
    # GPT
    # context = """import num"""
    # gpt = torch.load('model/tinygpt_shakespeare.pt')
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Char GPT
    context = """Title: Roger Federer"""
    gpt = torch.load('model/tinygpt_federer_char_small_model.pt')
    tokenizer = CharTokenizer()

    generate_text(gpt, tokenizer, 
                  context, 
                  top_p=0.95,
                  temperature=.5, 
                  max_len=2000,
                  device='mps')
