# Transformer Models in Natural Language Processing

A **transformer** is an encoder-decoder model that uses the attention mechanism. It can take advantage of parallelization and process a large amount of data simultaneously due to its model architecture. The attention mechanism is crucial for improving the performance of machine translation applications. Transformer models form the core of this architecture.
resize the image to 100x100

A typical transformer model consists of two main components: an **encoder** and a **decoder**. The encoder encodes the input sequence and passes it to the decoder. The encoder typically consists of multiple stacked encoders, and in the original research paper, six encoders were used. The number of encoders is a hyperparameter and not a magical number. Each encoder is structurally identical but has different weights. 

<div align="center">
  <img src="images\image.png" alt="Alt text" width="400" height="300" />
</div>

Each encoder can be divided into two sub-layers:
1. **Self-attention layer**: The input sequence flows through a self-attention layer, allowing the encoder to focus on relevant parts of the words as it encodes a central word in the input sentence.
2. **Feedforward layer**: The output of the self-attention layer is fed into a feedforward neural network. This network is applied independently to each position, making it suitable for parallel processing.

In the decoder, there are also self-attention and feedforward layers. Between them, there is an **encoder-decoder attention layer** that helps the decoder focus on relevant parts of the input sentence.

<div align="center">
  <img src="images\encoding.png" alt="Alt text" width="400" height="300" />
</div>

The self-attention layer breaks the input embedding into query, key, and value vectors. These vectors are computed using weights learned during training. Matrix computations are used to perform these operations in parallel. Once we have the query, key, and value vectors, the next step is to multiply each value vector by the softmax score in preparation to sum them up to calculate attention scores, allowing the model to focus on relevant words while suppressing irrelevant ones.

<div align="center">
  <img src="images\attention.png" alt="Alt text" width="400" height="300" />
</div>

In summary, the process of obtaining the final embeddings in a transformer involves:
1. Starting with a natural language sentence.
2. Embedding each word in the sentence.
3. Performing multi-headed attention multiple times,typically with eight heads, but this number can vary. Each attention head is like a different perspective through which the model can focus on different parts of the input sequence.
4. Multiplying the embedded words with weighted matrices.These matrices are unique for each attention head and capture how much attention should be given to each word with respect to the others. This step allows the model to weigh the importance of each word for the current context and the specific attention head.From the multiplication step, we obtain Query (Q), Key (K), and Value (V) matrices. These matrices are derived from the weighted embeddings and serve as the foundation for calculating attention scores.
<div style="display: flex; justify-content: center; align-items: center;">
  <div style="margin-right: 10px;">
    <img src="images\qkv1.png" alt="Alt text" width="350" height="200" />
  </div>
  <div style="margin-right: 10px;">
    <img src="images\qkv2.png" alt="Alt text" width="350" height="200" />
  </div>
  <div>
    <img src="images\qkv4.png" alt="Alt text" width="150" height="200" />
  </div>
</div>
5. Calculating attention using the resulting QKV (Query, Key, Value) matrices.These scores represent how much each word should attend to every other word in the input sequence. The scores are then scaled and passed through a softmax function to normalize them, ensuring that they sum up to 1 and can be interpreted as probabilities.
<div align="center">
  <img src="images\qkv3.png" alt="Alt text" width="300" height="200" />
</div>
6. Concatenating the matrices to produce the output matrix with the same dimension as the initial input.
<div align="center">
  <img src="images\concat.png" alt="Alt text" width="300" height="200" />
</div>

There are various variations of transformer models today. Some use both encoder and decoder components, some only use the encoder, and some only use the decoder. A popular encoder-only architecture is **BERT** (Bidirectional Encoder Representations from Transformers), developed by Google in 2018. BERT has different variations, including BERT base and BERT large, with different numbers of layers and parameters.

BERT is powerful because it can handle long input contexts. It was trained on a vast corpus of text, making it effective for various NLP tasks. BERT was trained on two tasks: masked language modeling and predicting the next sentence. These tasks allow BERT to work at both sentence and token levels.

To train BERT, three types of embeddings are used: token embeddings, segment embeddings (to distinguish between inputs in a pair), and position embeddings (to capture word order).

BERT can be applied to various downstream NLP tasks, including text classification, sentence-pair classification, question answering, and single-sentence tagging tasks.


# References
- [Transformer Models and BERT Model](https://www.coursera.org/learn/transformer-models-and-bert-model/home/info)






