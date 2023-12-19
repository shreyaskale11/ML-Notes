 





# 

Detailed Summary for [NeurIPS 2023 Poster Session 3 (Wednesday Evening)](https://www.youtube.com/watch?v=zn7nLR58hBk) by [Merlin](https://merlin.foyer.work/)

The title for the YouTube video could be: "Graph-Based Circuit Optimization for Pre-Layout Designing"

Solving the problem of pre-layout designing and constraint optimization.
- The process involves learning a graph, doing label propagation, generating a surrogate model, and constraint optimization.
- The approach reformulates the problem into a node regression problem in a semi-supervised learning framework.

Generating a graph representation from multiple feature vectors and labels
- Using optimization formulation to obtain weighted adjacency for the graph
- Incorporating directional energy, sparsity term, and regularizer to determine the graph structure

Using dynamic learning framework to converge towards optimal design parameters
- Using a convolutional neural network to learn a graph and predict labels for new solutions
- Implementing a dynamic learning framework with reranking and iteration to converge towards optimal design parameters

Using design parameters to satisfy specifications and minimize objective function
- Constructing features from design parameters to connect circuit instances into a graph
- Training on labeled samples using GNN and using ranker to generate more labeled samples

Differential Evolution algorithm is a heuristic algorithm for solving objective functions with unknown forms.
- The algorithm is designed to handle objective functions where traditional conditions may or may not work, and it aims to provide a faster solution.
- It involves selecting the best solution from different circuit instances and utilizing algorithms like Gat and Sage to generate new features and labels for previously unavailable samples.

Discussion on improving generalization ability in recommendation systems
- Exploration of parameter space and using D algorithm for label prediction
- Challenges and solutions for generalization with different testing and training distributions

Adverse training leads to better generalization abilities.
- Adverse training is equivalent to solving a distributional robust problem.
- It theoretically achieves better generalization and robustness, even in worst-case scenarios.

Knowledge distillation aims at creating similarity between student and teacher models.
- The goal is to ensure similarity between the outputs of the student and teacher model on given inputs.
- Spectral clustering helps the student model learn from the teacher model's good graph structure.

Using RKD for sample complexity and generalization guarantees in semi-supervised learning
- RKD as a regularization technique provides guarantees for sample complexity and generalization when the data is well-clustered.
- It can be applied in scenarios where a large teacher model can provide clustering structure and the student model can leverage RKD to learn from the pairwise similarities.

Evaluating similarity metrics for privacy leakage
- Different methods for measuring similarity and their limitations were discussed.
- The need for a universal and consistent metric for accurately identifying privacy leakage was emphasized.

Experimenting with different methods and networks for consistent performance.
- Switching from triplet loss to quive loss leads to improved results.
- Training with more samples shows consistent improvement.