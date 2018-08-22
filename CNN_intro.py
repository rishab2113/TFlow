# Initialization of weights options -

# Initialize all weights as zeros - No randomness
# Random distribution near zero - Activation function distortion

# Xavier Initialization - Optimal
# Draw weights from a distribution with zero mean and a specific variance

# Learning Rate - Defines step size during Gradient Descent

# Batch Size - Allow the use of Stochastic Gradient Descent -
# (Smaller - Less representative of data, Larger - Longer training time)

# Second-Order Behavior of Gradient Descent -
# Errors start off large, so larger steps at the beginning and smaller when moving closer to the minimum
# AdaGrad, RMSProp, ADAM (Best Optimizer, Automated Steps)

# Vanishing Gradients - Layers towards input will be affected less by error calculations -
# Initialization and Normalization help these issues

# Protection against Overfitting -
# L1/L2 Regularization (Penalty for larger weights)
# Dropout (Remove random neurons during training)
# Data Expansion (Artificially expand data by adding noise, tilt, white noise)

# MNIST Dataset
