"""
Generate text from GPT language model
"""
import torch
from transformers import GPT2TokenizerFast
from GPT import GPT

# TODO - slidding window text generation
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
        # TODO impelemnt nucleus decoding - top_p
        # greedy decoding
        pred_tokens = torch.argmax(logits, dim=-1).squeeze()    # (1, seq_len) 
        # index of last word before padding
        next_token_idx = attn_mask.to(torch.int32).sum() - 1                      
        next_token = pred_tokens[next_token_idx]

        pred_word = tokenizer.decode(next_token)
        show(pred_word, color=True)
        # auto-regressive generation
        context += pred_word

def show(text, color=False):
    c = e = ''
    if color:
        c = '\033[36m'
        e = '\033[0m'
    print(c + text + e, end='')
    

if __name__ == '__main__':
    context = """
Second Citizen:
One word, good"""
    gpt = torch.load('model/tinygpt.pt')
    generate_text(gpt, context, device='mps')
