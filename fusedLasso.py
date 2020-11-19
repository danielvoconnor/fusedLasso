# This code uses cvxpy to solve a fused Lasso problem.

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# First create some fake data.
m = 30
n = 50

betaTrue = np.zeros(n)
betaTrue[10:15] = 2.718
betaTrue[40:43] = -3.14159

np.random.seed(1)
X = np.random.randn(m, n)
noise = 0.01*np.random.randn(m)
y = X@betaTrue + noise 

# Now set regularization parameter values. 
lmbda = 10.0
gamma = 25.0

# Here's the part of the code where we use cvxpy to
# solve our fused Lasso problem.
beta = cp.Variable(n)
forwardDiff = beta[1:] - beta[0:-1]
objective = cp.Minimize(.5*cp.sum_squares(X@beta - y) + lmbda*cp.norm(beta,1) + gamma*cp.norm(forwardDiff,1))
constraints = [0 <= beta]
prob = cp.Problem(objective, constraints)
result = prob.solve()

# Plot the true and estimated beta vectors.
plt.figure()
plt.plot(betaTrue)
plt.plot(beta.value)
plt.title('betaTrue and estimated beta')
plt.legend(['betaTrue','estimated beta'])
plt.show()
