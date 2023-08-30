# tinyGPT
GPT-2 implementation using PyTorch


# TODO
- <s>Inference - generating tokens</s>
- <s>Reuse token embedding layer in the output layer</s>
- Pad token masking
- Update unit tests
- how to process text data? <bos>,<eos>,<pad>. How to split text file?
- Better data handling? Train on larger dataset.
- Track training loss - WnB

# Sub Projects
## Char LLM
Build char level LLM
* Char tokenizer - ASCII level
* Model pre-training
* Eval - perplexity / loss
* Model config
* Data loader

## Fine-tuning
Fine-tuning a pre-trained model


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

## Model Size
![gtp2 model size](https://jalammar.github.io/images/gpt2/gpt2-sizes-hyperparameters-3.png)
GPT-2 model sizes! No. of decoder stacks vs no. of parameters

![gpt2 parameter size](https://jalammar.github.io/images/gpt2/gpt2-sizes.png)
GPT-2 model parameter size

# Refs
[Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

[HF NLP course: Training a causual LM](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt)


https://rowlando13.medium.com/everything-gpt-2-4-data-preparation-514cb62f9f3b
https://kikaben.com/transformers-coding-details/

