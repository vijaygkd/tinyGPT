"""
Train GPT model
"""
from tqdm import tqdm
import torch
from torch import nn
from GPT import GPT
from dataset import GPTDataset, pad_seq_fn
from torch.utils.data import Dataset, DataLoader


def train():
    num_epochs = 1
    batch_size = 64
    seq_len = 128

    # dataset
    dataset = GPTDataset('data/tinyshakespeare.txt', seq_len=seq_len)
    vocab_size = dataset.tokenizer.vocab_size
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=pad_seq_fn)
    print(f"Dataset token size: {len(dataset)}")
    print(f"No. of batches: {len(dataloader)}")

    # model
    gpt = GPT(
        n_blocks=2,
        d_model=512,
        d_ff=512*4,
        n_heads=8,
        p_drop=0.1,
        vocab_size=vocab_size,
        seq_len=seq_len
    )

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.AdamW(gpt.parameters())  #optim.SGD(model.parameters(), lr=0.01) 

    print(gpt)
    print(f"Number of parameters: {sum(p.numel() for p in gpt.parameters())}")
    print("Starting training.")
    # train
    for epoch in range(num_epochs):
        gpt.train()

        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch')):
            X, Y = data
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
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print(f'Epoch {epoch}, batch {i+1}: loss = {running_loss / 10:.4f}')
                running_loss = 0.0

        # Print the loss for this epoch
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))



if __name__ == '__main__':
    train()