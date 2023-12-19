




# Neural Machine Translation

## Seq2seq Model

Google introduced a seq2seq model in 2014, which is a neural network that converts a sequence of symbols into another sequence of symbols. It is used in machine translation, text summarization, speech recognition, and more. In this notebook, we will build a neural machine translation model (NMT) with attention mechanism using TensorFlow. The model will translate English sentences into French sentences.

seq2seq model is a two recurrent neural network (RNN) models, an encoder and a decoder. The encoder reads an input sequence and outputs a single vector, and the decoder reads that vector to produce an output sequence. The encoder and decoder of seq2seq model are both RNNs, but they can be any other type of model such as convolutional neural networks (CNNs).

<div align="center">
  <img src="images/seq2seq_model.png" alt="Alt text" width="500" height="200" />
</div>

Seq2seq model shortcomings:
- The encoder of seq2seq model has to compress all the information of the source sentence into a single vector, which is difficult.
- The decoder of seq2seq model has to decode the entire target sentence from a single vector representation, which is difficult.
- The seq2seq model uses a fixed-length vector to encode a source sentence into a fixed-length vector, which is difficult.

Attention mechanism is a solution to the above shortcomings. It allows the decoder to look at the entire source sentence when it is decoding each word of the target sentence. It also allows the encoder to create a set of vectors, each representing a part of the source sentence, which is easier.

## Seq2seq Model with Attention Mechanism

Motivation of attention mechanism and how it works:
- The basic seq2seq model encodes the full input sequence into a fixed-length vector, the encoder's final hidden state. This vector is used as the initial hidden state for the decoder to generate the output sequence. The issue is it has to compress all the information of the input into this single vector.
- An improvement is to pass the encoder's hidden states at each timestep to the decoder instead of just the last one. But this requires keeping track of all the encoder hidden states in memory, which is inefficient.
- To address this, the encoder hidden states can be combined into a single context vector, typically by taking a weighted sum of the states. The weights allow the model to pay attention to different parts of the input at each timestep.
- The context vector is calculated as a weighted sum of the encoder hidden states. The weight given to each state is calculated by comparing it to the decoder's previous hidden state. This allows the decoder to focus on the most relevant encoder states for the next prediction.

The goal is to calculate a context vector $c_i$ that contains relevant information from the encoder hidden states $h_j$ for the $i^{th}$ decoding timestep. 

1. First, an alignment score $e_{ij}$ is calculated between the current decoder hidden state $s_{i-1}$ and each encoder hidden state $h_j$:

$$e_{ij} = f(s_{i-1}, h_j)$$ 

where $f$ is a feedforward network that learns to assign higher scores to encoder states that are relevant to predicting the next output word.

2. These scores are converted to probabilities (weights) using a softmax:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})}$$

where $T$ is the length of the input sequence. $k$ is the index of the encoder hidden state.

3. The context vector $c_i$ is then calculated as the weighted average of the encoder hidden states:

$$c_i = \sum_{j=1}^T \alpha_{ij} h_j$$

So the attention weights $\alpha_{ij}$ allow the model to focus on different parts of the input sequence by selectively weighting the encoder hidden states when constructing the context vector for each decoder output. The feedforward network learns what parts of the input to pay attention to.




## References
This course drew from the following resources:

Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
 (Raffel et al, 2019)

Reformer: The Efficient Transformer
 (Kitaev et al, 2020)

Attention Is All You Need
 (Vaswani et al, 2017)

Deep contextualized word representations
 (Peters et al, 2018)

The Illustrated Transformer
 (Alammar, 2018)

The Illustrated GPT-2 (Visualizing Transformer Language Models)
 (Alammar, 2019)
http://jalammar.github.io/illustrated-gpt2/ 

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
 (Devlin et al, 2018)

How GPT3 Works - Visualizations and Animations
 (Alammar, 2020)
http://jalammar.github.io/how-gpt3-works-visualizations-animations/ 
https://www.youtube.com/watch?v=rBCqOTEfxvg 


Word Embedding
https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca 

transformer_positional_encoding 
https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb 

Layer Normalization 
Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed- forward neural networks.
https://arxiv.org/pdf/1607.06450.pdf 



## Teacher Forcing

Teacher forcing is a method for quickly and efficiently training recurrent neural network models that use the ground truth from a prior time step as input.

You can use the ground truth words as decoder inputs instead of the decoder outputs. Even if the model makes a wrong prediction, it pretends as if it's made the correct one and this can continue. This method makes training much faster and has a special name, teacher forcing. 

There are some variations on this tool. For example, you can slowly start using decoder outputs over time, so that leads into training, you are no longer feeding in the target words. This is known as **curriculum learning**. You are now familiar with teacher forcing, and you can add this technique to your toolbox, to help you with training your model, and to help you get a better accuracy.


QKV history - 
Attention Is All You Need - https://arxiv.org/pdf/1706.03762.pdf 

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

Architecture of Extended Neural GPU [16], ByteNet [18] and ConvS2S [9],

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolu- tional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017. https://arxiv.org/pdf/1705.03122.pdf  
[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016. https://arxiv.org/pdf/1601.06733.pdf 
[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.
[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.
https://arxiv.org/pdf/1511.08228.pdf 
[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko- ray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017. https://arxiv.org/pdf/1610.10099.pdf
[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017. https://arxiv.org/pdf/1703.03130.pdf
[27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016. https://aclanthology.org/D16-1244.pdf 
[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.


https://bbycroft.net/llm 



## BLEU Score

BLEU is a metric for evaluating a generated sentence to a reference sentence. It is based on the idea that the closer a machine translation is to a professional human translation, the better it is.

method - 
1. Collect a set of reference sentences (e.g. human translations).
2. Collect a set of candidate sentences (e.g. machine translations).
3. For each candidate sentence, compare it to all the reference sentences and calculate a similarity score.
4. Calculate BLEU score for all the candidate sentences.

BLEU score is calculated as follows:
$$ BP = \begin{cases} 1 & \text{if } c>r \\ e^{(1-r/c)} & \text{if } c\leq r \end{cases} $$
$$ BLEU = BP \times \exp\left(\sum_{n=1}^N w_n \log p_n\right) $$

where $c$ is the length of the candidate sentence, $r$ is the length of the reference sentence, $p_n$ is the precision of $n$-grams, and $w_n$ is the weight of $n$-grams. The default weights are $w_n=1/N$.



