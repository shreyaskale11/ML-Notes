

The expression \(p(\mathbf{x}, \mathbf{y}) = p(\mathbf{x}) \cdot p(\mathbf{y}|\mathbf{x})\) represents the joint probability distribution of two random variables, \(\mathbf{x}\) and \(\mathbf{y}\). Let's break down what this expression means:

1. \(p(\mathbf{x})\): This represents the marginal probability distribution of \(\mathbf{x}\), meaning it describes the probability distribution of the random variable \(\mathbf{x}) independently, without considering the value of \(\mathbf{y}\). In other words, it tells you how likely different values of \(\mathbf{x}\) are without any reference to \(\mathbf{y}\).

2. \(p(\mathbf{y}|\mathbf{x})\): This is the conditional probability distribution of \(\mathbf{y}\) given \(\mathbf{x}\). It describes how the probability distribution of \(\mathbf{y}\) changes based on the value of \(\mathbf{x}\). In essence, it tells you how likely different values of \(\mathbf{y}\) are when you already know the value of \(\mathbf{x}\).

3. \(p(\mathbf{x}, \mathbf{y})\): This is the joint probability distribution of both \(\mathbf{x}\) and \(\mathbf{y}\). It describes how likely it is to observe a specific combination of values for both variables together.

The expression \(p(\mathbf{x}, \mathbf{y}) = p(\mathbf{x}) \cdot p(\mathbf{y}|\mathbf{x})\) is based on the product rule of probability. It tells us that the joint probability of \(\mathbf{x}\) and \(\mathbf{y}\) can be calculated by multiplying the marginal probability of \(\mathbf{x}\) and the conditional probability of \(\mathbf{y}\) given \(\mathbf{x}\).

In practical terms, this expression is often used in probability and statistics to model and reason about the joint distribution of two variables in terms of their marginal and conditional probabilities.

Let's use an example to illustrate the concept of \(p(x, y) = p(x) \cdot p(y | x)\) in the context of probability.

**Example: Coin Toss and Weather**

Suppose we have two random variables, \(x\) representing the outcome of a coin toss (Heads or Tails), and \(y\) representing the weather condition (Sunny, Rainy, or Cloudy) in a particular location.

We want to model the joint probability distribution of the coin toss and the weather, meaning we want to understand how likely it is to observe specific combinations of these events.

Here's how we can apply \(p(x, y) = p(x) \cdot p(y | x)\) to this example:

1. **Marginal Probability of Coin Toss (\(p(x)\)):**
   - \(p(x = \text{Heads}) = 0.5\) (Assuming a fair coin, Heads and Tails are equally likely).
   - \(p(x = \text{Tails}) = 0.5\)

2. **Conditional Probability of Weather (\(p(y | x)\)):**
   - If the coin lands on Heads (\(x = \text{Heads}\)), the probability of Sunny weather is \(p(y = \text{Sunny} | x = \text{Heads}) = 0.7\) (70% chance of being sunny after a Heads).
   - If the coin lands on Heads (\(x = \text{Heads}\)), the probability of Rainy weather is \(p(y = \text{Rainy} | x = \text{Heads}) = 0.2\) (20% chance of being rainy after a Heads).
   - If the coin lands on Heads (\(x = \text{Heads}\)), the probability of Cloudy weather is \(p(y = \text{Cloudy} | x = \text{Heads}) = 0.1\) (10% chance of being cloudy after a Heads).
   - Similarly, we can define conditional probabilities for Tails (\(x = \text{Tails}\)).

Now, let's use the formula:

\(p(x, y) = p(x) \cdot p(y | x)\)

- For the combination of Heads (\(x = \text{Heads}\)) and Sunny weather (\(y = \text{Sunny}\)):
  - \(p(x = \text{Heads}) \cdot p(y = \text{Sunny} | x = \text{Heads}) = 0.5 \cdot 0.7 = 0.35\)

- For the combination of Tails (\(x = \text{Tails}\)) and Rainy weather (\(y = \text{Rainy}\)):
  - \(p(x = \text{Tails}) \cdot p(y = \text{Rainy} | x = \text{Tails}) = 0.5 \cdot 0.3 = 0.15\)

This calculation allows us to find the joint probability of specific combinations of the coin toss and the weather. It essentially breaks down the joint probability into the product of the marginal probability of the coin toss and the conditional probability of the weather given the coin toss outcome.

In practice, this formula is a fundamental concept in probability theory and is used to model more complex scenarios where events are dependent on each other. It helps us understand how the joint probability distribution of two variables can be expressed in terms of their marginal and conditional probabilities.