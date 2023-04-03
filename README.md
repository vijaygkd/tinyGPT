# tinyGPT
GPT-2 implementation using PyTorch


# TODO
- Pad token masking
- Inference - generating tokens
- Update unit tests
- how to process text data? <bos>,<eos>,<pad>. How to split text file?
- Better data handling? Train on larger dataset.
- Reuse token embedding layer in the output layer
- Track training loss


# Architecture
## Model
A. Transformer architecture
0. Build Attention components
    - Scaled dot-product attention
    - Multi-head attention
    - Position wise FF network
    - Masking
1. Build Embeddings
    - Token embeddings
    - position encoding
2. Build Encoder
    - Emcoder stack
3. Build Decoder
    - Decoder stack
    - Softmax output

## Data
1. Data Loader class
2. Masking
    - padding mask
    - subsequent mask


# Refs
https://rowlando13.medium.com/everything-gpt-2-4-data-preparation-514cb62f9f3b

