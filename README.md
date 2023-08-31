# tinyGPT
GPT-2 implementation using PyTorch


# TODO
- <s>Inference - generating tokens</s>
- <s>Reuse token embedding layer in the output layer</s> Need to test if this does help or not.
- <s>Track training loss - WnB</s>
- Pad token masking. Currently the way data is prepared for training, there is no padding.
- Update unit tests
- how to prepare data for pre-training? 
    - How to split text into sequence lengths and batches? Is it done randomly?
    - How are `<bos>,<eos>,<pad>` tokens used?
    - How are larger datasets combined?

## Pre-training models
How to use pre-trained open source models like Llama-2?

## Fine-tuning
Fine-tuning a pre-trained model.
* Finetune on custom dataset
* Try Instruct-finetuning
* Alpaca instruct-finetuning datasets
* How many epochs to fine-tune? Usualy its 1-2 epochs.


# Notes
## Learning rate

Transformers are sensitive to optimizer learning rate. It is one of most important hyper-parameter in training transformers.

* The transformer models converge at lower learning rates: **`lr < 1e-5`**. When transformer model is converging, the loss continously keeps dropping and can get very close to zero.
* At higher lr, loss doesn't drop and stagnates indicating the model is not learning.
* Models converge on smaller dataset like (data/rogerfederer.txt) at `5e-5`. Loss drops to almost `0` and the dataset is memorized by the model.
* TODO - The model training could benefit from learning rate schedulers often used in literature.


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

## Training
### GPT
Use configs from GPT paper.
* Byte pair encoding

### Char LLM
Build char level LLM
* Char tokenizer - ASCII level
* Model pre-training


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

