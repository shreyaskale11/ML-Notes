# Navie Bayes

## Introduction to Navie Bayes

Navie Bayes is a classification algorithm based on Bayes' theorem. According to Bayes' theorem, the conditional probability of an event A, given another event B, is given by P(A|B) = P(B|A)P(A)/P(B). In the context of classification, we can think of P(A) as the prior probability of A, and P(A|B) as the posterior probability of A. In other words, we can think of the posterior probability as the probability of A given the data B.

## Conditional Probability

Conditional probability is the probability of an event given that another event has occurred. For example, let's say that we have two events A and B. The probability of A given that B has occurred is given by:

$$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$
where P(A ∩ B) is the probability of A and B occurring together.

## Bayes' Rule:

Probability of A given B is equal to the probability of interestion of A and B divided by the probability of B.
$$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$
Similarly, we can write the probability of B given A as:
$$ P(B|A) = \frac{P(A \cap B)}{P(A)} $$

If we rearrange the above equation, we get:
$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

In the context of classification, we can think of P(A) as the prior probability of A, and P(A|B) as the posterior probability of A. In other words, we can think of the posterior probability as the probability of A given the data B.

## Navie Bayes Classifier

For tweets classification, we can use Navie Bayes classifier to classify tweets into positive and negative tweets. 

product of conditional probability of words in tweet of positive class and negative class:
$$ P(positive|tweet) = P(w_1|positive) * P(w_2|positive) * P(w_3|positive) * ... * P(w_n|positive) $$
$$ P(negative|tweet) = P(w_1|negative) * P(w_2|negative) * P(w_3|negative) * ... * P(w_n|negative) $$
where w1, w2, w3, ... , wn are words in tweet.
Calculate likelihood of tweet being positive and negative:
$$
\frac{P(\text{positive}|\text{tweet})}{P(\text{negative}|\text{tweet})} = 
\begin{cases} >1, & \text{positive} \\ <1, & \text{negative} \end{cases}
$$

## Laplacian Smoothing

We usually compute the probability of a word given a class as follows:

$$
P(w_i|\text{class}) = \frac{\text{freq}(w_i, \text{class})}{N_{\text{class}}} \qquad \text{class} \in \{\text{Positive}, \text{Negative}\} 
$$

However, if a word does not appear in the training, then it automatically gets a probability of 0, to fix this we add smoothing as follows

$$
P(w_i|\text{class}) = \frac{\text{freq}(w_i, \text{class}) + 1}{(N_{\text{class}} + V)}
$$

Note that we added a 1 in the numerator, and since there are $V$ words to normalize, we add $V$ in the denominator.

$N_{\text{class}}$: number of words in class

$V$: number of unique words in vocabulary

## Log Likelihood

We can use log likelihood to avoid underflow. The log likelihood is given by:

$$ \lambda(w) = \log \frac{P(w|\text{pos})}{P(w|\text{neg})} $$

where $P(w|\text{pos})$ and $P(w|\text{neg})$ are computed using Laplacian smoothing.


## train naïve Bayes classifier

1) Get or annotate a dataset with positive and negative tweets

2) Preprocess the tweets:

    Lowercase, Remove punctuation, urls, names , Remove stop words , Stemming , Tokenize sentences
    
3) Compute $\text{freq}(w, \text{class})$:

4) Get $P(w|\text{pos}), P(w|\text{neg})$

5) Get $\lambda(w)$

$$
\lambda(w) = \log \frac{P(w|\text{pos})}{P(w|\text{neg})}
$$

6) Compute $\text{logprior}$

$$
\text{logprior} = \log \frac{D_{\text{pos}}}{D_{\text{neg}}}
$$

where $D_{\text{pos}}$ and $D_{\text{neg}}$ correspond to the number of positive and negative documents respectively.

7) Compute Score 
$$\text{logprior} + \sum_{w \in \text{tweet}} \lambda(w) = 
\begin{cases} >1, & \text{positive} \\ <1, & \text{negative} \end{cases}
$$

