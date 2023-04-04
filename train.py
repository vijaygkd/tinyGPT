"""
Train GPT model
"""
from tqdm import tqdm
import torch
from torch import nn
from GPT import GPT
from dataset import GPTDataset, pad_seq_fn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary


def train():
    # ---------------------- #
    # MODEL PARAMETERS #
    n_blocks = 4
    d_model = 512
    d_ff = d_model * 4
    n_heads = 8
    p_drop = 0.1

    # TRAINING PARAMETERS #
    num_epochs = 50
    batch_size = 32
    seq_len = 50
    lr=0.001    # default=0.001
    model_path = 'model/tinygpt_ws.pt'
    data_path = 'data/tinyshakespeare.txt'

    # ---------------------------------------- #

    device = torch.device('mps' if torch.has_mps else 'cpu')
    # device = 'cpu'
    print(f"Hardware: {device}")

    # dataset
    dataset = GPTDataset(data_path, seq_len=seq_len)
    vocab_size = dataset.tokenizer.vocab_size
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=pad_seq_fn)
    num_batches = len(dataloader)
    print(f"Dataset token size: {len(dataset)}")
    print(f"No. of batches: {num_batches}")

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
    if True:
        gpt = torch.load(model_path)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=lr)

    summary(gpt, (batch_size, seq_len), dtypes=[torch.long], depth=3, device=device)
    print("Starting training.")
    # train
    for epoch in range(num_epochs):
        gpt.train()

        running_loss = 0.0
        with tqdm(total=num_batches, desc=f'Epoch {epoch}', unit='batch') as pbar:
            # for i, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch')):
            for i, data in enumerate(dataloader):
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

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # print statistics
                pbar.set_postfix(loss=loss.item())
                pbar.update()

                running_loss += loss.item()
                if i % 10 == 9:    # print every 10 mini-batches
                    # print(f'Epoch {epoch}, batch {i+1}: loss = {running_loss / 10:.4f}')
                    # Update the tqdm progress bar with the current loss value
                    # pbar.set_postfix(loss=f'{running_loss / 10:.4f}')
                    running_loss = 0.0

        # Print the loss for this epoch
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

    # save model
    torch.save(gpt, model_path)


if __name__ == '__main__':
    train()