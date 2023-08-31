"""
Train GPT model
"""
from tqdm import tqdm
import wandb
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from transformers import GPT2TokenizerFast
from GPT import GPT
from dataset import GPTDataset, CharTokenizer, pad_seq_fn


def train(model_path, tokenizer, collate_fn, dataset_train, dataset_val=None, log_wandb=True):
    # ---------------------- #
    # MODEL PARAMETERS #
    # GPT-2 small parameters
    n_blocks = 12     #12
    d_model = 512    #768
    d_ff = d_model * 4
    n_heads = 8      #12
    p_drop = 0.1      #0.1
    seq_len = dataset_train.seq_len
    vocab_size = tokenizer.vocab_size      # GPT2 tokenizer vocab size or char vocab size

    # TRAINING PARAMETERS #
    epochs = 20
    batch_size = 32
    lr = 5e-6       # transformers converge at low learning rates. At higher learning rate > 1e-4 loss doesn't drop. 
    # TODO - learning rate scheduler

    # wandb init
    if log_wandb:
        wandb.init(
            project="tiny-gpt",
            name=model_path.split('/')[-1],
            config={
                "n_blocks": n_blocks,
                "d_model": d_model,
                "d_ff": d_ff,
                "n_heads": n_heads,
                "p_drop": p_drop,
                "vocab_size": vocab_size,
                "seq_len": seq_len,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "model_path": model_path,
            }
        )

    # ---------------------------------------- #


    device = torch.device('mps' if torch.has_mps else 'cpu')
    device = torch.device('cuda' if torch.has_cuda else device)
    # device = 'cpu'
    print(f"Hardware: {device}")

    # dataset
    dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=collate_fn)
    if dataset_val:
        dataloader_val = DataLoader(dataset_val, batch_size, shuffle=True, collate_fn=collate_fn)
    num_batches = len(dataloader_train)
    print(f"No. of batches: {num_batches}")
    print(f"Train Dataset token size: {num_batches * batch_size * seq_len}")

    # model
    gpt = GPT(
        n_blocks=n_blocks,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        p_drop=p_drop,
        vocab_size=vocab_size,
        seq_len=seq_len,
        device=device
    )

    # load existing
    if False:
        gpt = torch.load(model_path)

    # print model summary
    summary(gpt, (batch_size, seq_len), dtypes=[torch.long], depth=3, device=device)

    # torch 2.0 accelerator -> only works with CUDA
    if torch.cuda.is_available():
        gpt = torch.compile(gpt)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=lr)
    
    print("Starting training.")
    # train
    for epoch in range(epochs):
        gpt.train()

        with tqdm(total=num_batches, desc=f'Epoch {epoch}', unit='batch') as pbar:
            # for i, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch')):
            for i, data in enumerate(dataloader_train):
                X, Y = data
                X, Y = X.to(device), Y.to(device)
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                logits, attns = gpt(X)
                #y_pred = torch.argmax(logits, dim=-1)
                logits = logits.view(-1, vocab_size)
                targets = Y.view(-1)
                
                # Compute the loss
                loss = criterion(logits, targets)
                perplexity = torch.exp(loss)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # print statistics
                pbar.set_postfix(loss=loss.item(), perplexity=perplexity.item())
                pbar.update()

                # wandb
                wandb.log({"loss": loss.item(), "perplexity": perplexity.item()})

                # running_loss += loss.item()
                # running_ppx += perplexity.item()
                # if i % 10 == 9:    # print every 10 mini-batches
                #     # print(f'Epoch {epoch}, batch {i+1}: loss = {running_loss / 10:.4f}')
                #     # Update the tqdm progress bar with the current loss value
                #     # pbar.set_postfix(loss=f'{running_loss / 10:.4f}')
                #     running_loss = 0.0
                #     running_ppx = 0.0
                # TODO - run eval on validation data

    # save model
    torch.save(gpt, model_path)
    wandb.finish()


def train_codeparrot():
    from dataset_hf import get_codeparrot_dataset, prepare_input_labels

    print("Training Codeparrot dataset.")
    model_path = 'model/tinygpt_codeparrot.pt'
    tkz = GPT2TokenizerFast.from_pretrained('gpt2')
    dataset_train = get_codeparrot_dataset(seq_len=128, split='valid')
    # dataset_val = get_codeparrot_dataset(seq_len=128, split='valid')  
    collate_fn = prepare_input_labels
    # train model
    train(model_path, tkz, collate_fn, dataset_train, None)


def train_shakespeare():
    from dataset import GPTDataset, pad_seq_fn

    print("Training Shakespeare dataset.")
    data_path = 'data/tinyshakespeare.txt'
    model_path = 'model/tinygpt_shakespeare_bpe_small.pt'

    tkz = GPT2TokenizerFast.from_pretrained('gpt2')
    dataset = GPTDataset(tkz, data_path, seq_len=256)
    collate_fn=lambda x: pad_seq_fn(x, tkz.eos_token_id)
    # train model
    train(model_path, tkz, collate_fn, dataset)


def train_shakespeare_char():
    from dataset import GPTDataset, pad_seq_fn

    print("Training Shakespeare dataset at Character level.")
    data_path = 'data/tinyshakespeare.txt'
    model_path = 'model/tinygpt_shakespeare_char_small_model_lowlr.pt'

    tkz = CharTokenizer()
    dataset = GPTDataset(tkz, data_path, seq_len=512)
    collate_fn=lambda x: pad_seq_fn(x, tkz.eos_token_id)
    # train model
    train(model_path, tkz, collate_fn, dataset)


def train_federer_char():
    from dataset import GPTDataset, pad_seq_fn

    print("Training Federer dataset at Character level.")
    data_path = 'data/rogerfederer.txt'
    model_path = 'model/tinygpt_federer_char_small_model.pt'

    tkz = CharTokenizer()
    dataset = GPTDataset(tkz, data_path, seq_len=512)
    collate_fn=lambda x: pad_seq_fn(x, tkz.eos_token_id)
    # train model
    train(model_path, tkz, collate_fn, dataset)



if __name__ == '__main__':
    # train_codeparrot()
    # train_shakespeare()
    train_shakespeare_char()
    # train_federer_char()
