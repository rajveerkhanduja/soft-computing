# soft-computing

# Write a Python program to implement all steps of GA (selection, crossover,

import random

fitness function
def fitness(x):
    return x * x        maximize x²

initial population
pop = [random.uniform(-10, 10) for _ in range(5)]

for gen in range(10):
    new_pop = []

    for _ in range(5):
        selection
        p1 = random.choice(pop)
        p2 = random.choice(pop)

        crossover
        child = (p1 + p2) / 2

        mutation
        child += random.uniform(-1, 1)

        new_pop.append(child)

    pop = new_pop
    best = max(pop, key=fitness)
    print("Gen", gen+1, " Best:", best)

# Implement GA to optimize weights of a simple equation.
import random

X = [1,2,3]
Y = [2,4,6]

def fitness(w):
    error = 0
    for i in range(3):
        error += (Y[i] - w * X[i])**2
    return -error    # maximize negative error

pop = [random.uniform(0,5) for _ in range(5)]

for gen in range(10):
    new = []
    for _ in range(5):
        p1 = random.choice(pop)
        p2 = random.choice(pop)
        child = (p1 + p2) / 2
        child += random.uniform(-0.5,0.5)
        new.append(child)
    pop = new
    best = max(pop, key=fitness)
    print("Gen", gen+1, " w =", best)

# Use GA to solve a simple optimization problem (maximize f(x) or minimize error).

import random
import math

def f(x):
    return x + math.sin(x)

pop = [random.uniform(-10,10) for _ in range(5)]

for gen in range(10):
    new = []
    for _ in range(5):
        p1, p2 = random.choice(pop), random.choice(pop)
        child = (p1 + p2) / 2
        child += random.uniform(-1,1)
        new.append(child)
    pop = new
    best = max(pop, key=f)
    print("Gen", gen+1, " x =", best)

# Write a Python program to implement addition, subtraction, multiplication of fuzzy numbers.

Triangular fuzzy number arithmetic

def add(A, B):
    return (A[0]+B[0], A[1]+B[1], A[2]+B[2])

def sub(A, B):
    return (A[0]-B[0], A[1]-B[1], A[2]-B[2])

def mul(A, B):
    return (A[0]*B[0], A[1]*B[1], A[2]*B[2])

A = (1, 2, 3)
B = (2, 3, 4)

print("Addition:", add(A, B))
print("Subtraction:", sub(A, B))
print("Multiplication:", mul(A, B))

# 44. Implement triangular fuzzy number arithmetic and plot membership functions.
import numpy as np
import matplotlib.pyplot as plt

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x-a)/(b-a), (c-x)/(c-b)), 0)

x = np.linspace(0, 10, 100)
A = (2, 4, 6)
B = (1, 3, 5)

plt.plot(x, trimf(x, *A), label="A")
plt.plot(x, trimf(x, *B), label="B")
plt.legend()
plt.show()

# Implement fuzzy α-cut operations for interval arithmetic.
def alpha_cut(A, alpha):
    a, b, c = A
    left = a + alpha*(b-a)
    right = c - alpha*(c-b)
    return (left, right)

A = (2, 4, 8)
print(alpha_cut(A, 0.5))


# Demonstrate the effect of fuzzy scaling (multiplying fuzzy sets by crisp values).
def scale(A, k):
    return (A[0]*k, A[1]*k, A[2]*k)

A = (1, 3, 5)
k = 2
print(scale(A, k))


# 53. Write a Python program to implement fuzzy relation matrix and compute max-min composition.
import numpy as np

R1 = np.array([[0.2, 0.7],
               [0.5, 1.0]])

R2 = np.array([[0.6, 0.1],
               [0.9, 0.4]])

Max-min composition
result = np.zeros((2,2))

for i in range(2):
    for j in range(2):
        result[i][j] = np.max(np.minimum(R1[i], R2[:,j]))

print(result)

# Evaluate reflexivity, symmetry, and transitivity of a fuzzy relation.
import numpy as np

R = np.array([[1.0, 0.5],
              [0.5, 1.0]])

Reflexive: diagonal must be 1
print("Reflexive:", np.all(np.diag(R) == 1))

Symmetric: R[i][j] == R[j][i]
print("Symmetric:", np.all(R == R.T))

Transitive (Max-Min)
T = np.zeros_like(R)
for i in range(2):
    for j in range(2):
        T[i][j] = np.max(np.minimum(R[i], R[:,j]))

print("Transitive:", np.all(T <= R))

# Implement fuzzy equivalence relation closure for a given dataset.
import numpy as np

R = np.array([[1.0, 0.4],
              [0.2, 1.0]])

Reflexive closure
for i in range(len(R)):
    R[i][i] = 1

Symmetric closure
R = np.maximum(R, R.T)

Transitive closure (Warshall-type)
n = len(R)
for k in range(n):
    for i in range(n):
        for j in range(n):
            R[i][j] = max(R[i][j], min(R[i][k], R[k][j]))

print(R)


# Write a Python program to implement centroid, and MOM defuzzification.
import numpy as np

x = np.linspace(0, 10, 100)
mu = np.array([min(i/10, (10-i)/10) for i in x])

# Centroid
centroid = np.sum(x * mu) / np.sum(mu)

# MOM
mom = x[np.argmax(mu)]

print("Centroid:", centroid)
print("MOM:", mom)

# Compare outputs of three defuzzifiers using the same fuzzy inference system and analyze differences.
import numpy as np

x = np.linspace(0, 10, 100)
mu = np.maximum(0, 1 - abs(x-5)/3)

# Centroid
centroid = np.sum(x * mu) / np.sum(mu)

# MOM
mom = x[np.argmax(mu)]

# BOA (mean of all maxima)
max_val = np.max(mu)
boa = np.mean(x[mu == max_val])

print("Centroid:", centroid)
print("MOM:", mom)
print("BOA:", boa)

# Write a Python program to implement a perceptron to classify students as Pass/Fail using semester marks.
# perceptron_binary.py
import numpy as np

# synthetic data: columns = [sem1, sem2], rows = students
X = np.array([[40,50],[60,70],[30,20],[80,85],[45,44],[55,52]])
y = np.array([0,1,0,1,0,1])   # 1 = Pass, 0 = Fail

# add bias term
Xb = np.hstack([np.ones((X.shape[0],1)), X])

w = np.zeros(Xb.shape[1])     # initialize weights (bias included)
lr = 0.01
epochs = 50

for epoch in range(epochs):
    for xi, target in zip(Xb, y):
        out = 1 if np.dot(w, xi) >= 0 else 0
        error = target - out
        w += lr * error * xi

# predict
pred = np.array([1 if np.dot(w, xi) >= 0 else 0 for xi in Xb])
print("Weights:", w)
print("True:", y)
print("Pred:", pred)
print("Accuracy:", (pred==y).mean())


# Modify the perceptron to classify students into three categories (Poor, Average, Good).

# perceptron_multiclass.py
import numpy as np

# labels: 0=Poor,1=Average,2=Good
X = np.array([[30,35],[50,55],[80,85],[45,50],[60,62],[20,25]])
y = np.array([0,1,2,1,2,0])

Xb = np.hstack([np.ones((X.shape[0],1)), X])
num_classes = 3
W = np.zeros((num_classes, Xb.shape[1]))  # each row = weights for one class
lr = 0.01
epochs = 100

for epoch in range(epochs):
    for xi, target in zip(Xb, y):
        scores = W.dot(xi)
        pred = np.argmax(scores)
        if pred != target:
            W[target] += lr * xi
            W[pred] -= lr * xi

preds = np.argmax(W.dot(Xb.T), axis=0)
print("True:", y)
print("Pred:", preds)
print("Accuracy:", (preds==y).mean())

# Implement perceptron learning with different learning rates and compare convergence.

# perceptron_lr_compare.py
import numpy as np

X = np.array([[40,50],[60,70],[30,20],[80,85],[45,44],[55,52]])
y = np.array([0,1,0,1,0,1])
Xb = np.hstack([np.ones((X.shape[0],1)), X])

def train(lr, max_epochs=100):
    w = np.zeros(Xb.shape[1])
    for epoch in range(1, max_epochs+1):
        errors = 0
        for xi, target in zip(Xb,y):
            out = 1 if w.dot(xi) >= 0 else 0
            err = target - out
            if err != 0:
                w += lr * err * xi
                errors += 1
        if errors == 0:
            return epoch, w
    return None, w  # didn't fully converge

for lr in [0.0001, 0.001, 0.01, 0.1, 0.5]:
    epoch_converged, w = train(lr)
    print(f"lr={lr:6} -> converged in epoch {epoch_converged}, final weights {w}")

# Train a perceptron using normalized semester marks and analyze accuracy differences.

# perceptron_normalized.py
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.array([[40,50],[60,70],[30,20],[80,85],[45,44],[55,52]])
y = np.array([0,1,0,1,0,1])
Xb = np.hstack([np.ones((X.shape[0],1)), X])

# train on raw
w_raw = np.zeros(Xb.shape[1])
lr = 0.01
for _ in range(100):
    for xi, t in zip(Xb,y):
        out = 1 if w_raw.dot(xi) >= 0 else 0
        w_raw += lr * (t - out) * xi

# train on normalized features (no bias normalized; keep bias separately)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
Xb_norm = np.hstack([np.ones((X.shape[0],1)), X_norm])
w_norm = np.zeros(Xb_norm.shape[1])
for _ in range(100):
    for xi, t in zip(Xb_norm,y):
        out = 1 if w_norm.dot(xi) >= 0 else 0
        w_norm += lr * (t - out) * xi

pred_raw = np.array([1 if w_raw.dot(xi)>=0 else 0 for xi in Xb])
pred_norm = np.array([1 if w_norm.dot(xi)>=0 else 0 for xi in Xb_norm])
print("Accuracy raw:", (pred_raw==y).mean())
print("Accuracy normalized:", (pred_norm==y).mean())


# Implement perceptron with bias = 0 and bias ≠ 0; observe output.
# perceptron_bias_compare.py
import numpy as np

X = np.array([[40,50],[60,70],[30,20],[80,85]])
y = np.array([0,1,0,1])

# case A: bias = 0 (no bias term)
W_nobias = np.zeros(X.shape[1])
lr=0.01
for _ in range(100):
    for xi,t in zip(X,y):
        out = 1 if W_nobias.dot(xi) >= 0 else 0
        W_nobias += lr * (t - out) * xi

# case B: bias included
Xb = np.hstack([np.ones((X.shape[0],1)), X])
W_bias = np.zeros(Xb.shape[1])
for _ in range(100):
    for xi,t in zip(Xb,y):
        out = 1 if W_bias.dot(xi) >= 0 else 0
        W_bias += lr * (t - out) * xi

print("No-bias weights:", W_nobias)
print("With-bias weights:", W_bias)
print("Pred no-bias:", [1 if W_nobias.dot(x)>=0 else 0 for x in X])
print("Pred with-bias:", [1 if W_bias.dot(xi)>=0 else 0 for xi in Xb])

# Plot decision boundary for a perceptron trained on two subjects’ marks.
# perceptron_decision_boundary.py
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[30,40],[45,55],[60,70],[25,20],[70,80],[50,30]])
y = np.array([0,1,1,0,1,0])
Xb = np.hstack([np.ones((X.shape[0],1)), X])

w = np.zeros(Xb.shape[1])
lr=0.01
for _ in range(200):
    for xi,t in zip(Xb,y):
        out = 1 if w.dot(xi)>=0 else 0
        w += lr*(t-out)*xi

# plot
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k')
xmin, xmax = X[:,0].min()-5, X[:,0].max()+5
xs = np.linspace(xmin, xmax, 100)
# decision boundary: w0 + w1*x + w2*y = 0 -> y = -(w0 + w1*x)/w2
if abs(w[2]) < 1e-6:
    print("Vertical boundary (w2 ~= 0). Weights:", w)
else:
    ys = -(w[0] + w[1]*xs) / w[2]
    plt.plot(xs, ys, 'g-')
plt.xlabel("Subject1"); plt.ylabel("Subject2"); plt.title("Decision boundary")
plt.show()


# Extend the perceptron to handle multi-subject marks (5 subjects) using vectorized inputs.

# perceptron_vector_5subjects.py
import numpy as np

# 5 subjects per student
X = np.array([
    [40,42,38,50,45],
    [60,62,65,58,59],
    [30,28,25,35,32],
    [75,78,80,70,72],
    [50,48,52,49,51]
])
y = np.array([0,1,0,1,1])  # binary pass/fail

# vectorized perceptron training (batch updates)
Xb = np.hstack([np.ones((X.shape[0],1)), X])
w = np.zeros(Xb.shape[1])
lr = 0.01
epochs = 100

for epoch in range(epochs):
    outputs = (Xb.dot(w) >= 0).astype(int)
    errors = y - outputs
    # batch weight update: w += lr * sum(errors_i * x_i)
    w += lr * (errors @ Xb)
print("Weights:", w)
pred = (Xb.dot(w) >= 0).astype(int)
print("Accuracy:", (pred==y).mean())


# Implement perceptron weight updates step-by-step and print weight changes after each epoch.
# perceptron_print_updates.py
import numpy as np

X = np.array([[30,40],[50,60],[20,25]])
y = np.array([0,1,0])
Xb = np.hstack([np.ones((X.shape[0],1)), X])

w = np.zeros(Xb.shape[1])
lr = 0.05
epochs = 10

for epoch in range(1, epochs+1):
    print(f"Epoch {epoch} start weights: {w}")
    for i, (xi, target) in enumerate(zip(Xb, y), start=1):
        out = 1 if w.dot(xi) >= 0 else 0
        error = target - out
        delta = lr * error * xi
        w += delta
        print(f"  Sample {i}: target={target}, out={out}, delta={delta}, new_w={w}")
    print(f"Epoch {epoch} end weights: {w}\n")


# Write a Python program to implement a 2-layer Feed Forward Network for semester-mark classification.
# ffn_two_layer.py
import numpy as np

# Data: [sem1, sem2] → Pass/Fail
X = np.array([[40,50],[60,70],[30,20],[80,85]])
y = np.array([0,1,0,1])

# weights: 2 inputs → 1 output
W = np.random.randn(2)
b = 0.0
lr = 0.001

def step(z):
    return 1 if z >= 0 else 0

for epoch in range(200):
    for xi, target in zip(X, y):
        z = np.dot(W, xi) + b
        out = step(z)
        error = target - out
        W += lr * error * xi
        b += lr * error

print("Final weights:", W, "bias:", b)
print("Predictions:", [step(np.dot(W, xi) + b) for xi in X])


# Extend the network to include a hidden layer with sigmoid activation implemented manually.
# ffn_hidden_sigmoid.py
import numpy as np

X = np.array([[30,40],[50,60],[20,25],[70,75]])
y = np.array([0,1,0,1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sizes: 2 inputs → 3 hidden → 1 output
W1 = np.random.randn(2,3)
b1 = np.zeros(3)
W2 = np.random.randn(3)
b2 = 0.0
lr = 0.01

for epoch in range(500):
    for xi, target in zip(X,y):
        h = sigmoid(xi @ W1 + b1)
        o = sigmoid(h @ W2 + b2)
        error = target - o
        
        # backprop (manual)
        dO = o * (1 - o) * error
        dW2 = dO * h
        db2 = dO

        dH = dO * W2 * h * (1 - h)
        dW1 = np.outer(xi, dH)
        db1 = dH

        # update
        W2 += lr * dW2
        b2 += lr * db2
        W1 += lr * dW1
        b1 += lr * db1

print("Predictions:", [sigmoid(sigmoid(xi@W1+b1)@W2+b2) for xi in X])

# Implement forward propagation using matrix multiplication.
# forward_matrix.py
import numpy as np

X = np.array([[30,40],[50,60],[20,25]])
y = np.array([[0],[1],[0]])

def sigmoid(x):
    return 1/(1+np.exp(-x))

# 2 → 3 → 1 network
W1 = np.random.randn(2,3)
b1 = np.zeros((1,3))
W2 = np.random.randn(3,1)
b2 = np.zeros((1,1))

# forward pass (FULLY VECTORIZED)
H = sigmoid(X @ W1 + b1)
O = sigmoid(H @ W2 + b2)

print("Output:", O)

# Compare performance of TANH and ReLU activation functions in the feed- forward network.

# tanh_relu_compare.py
import numpy as np

X = np.array([[30,40],[50,60],[20,25],[70,75]])
y = np.array([0,1,0,1])

def tanh(x):  return np.tanh(x)
def relu(x):  return np.maximum(0, x)
def sigmoid(x): return 1/(1+np.exp(-x))

def train(act):
    W = np.random.randn(2,3)
    b = np.zeros(3)
    W2 = np.random.randn(3)
    b2 = 0.0
    lr = 0.001

    for _ in range(300):
        for xi, target in zip(X,y):
            h = act(xi @ W + b)
            o = sigmoid(h @ W2 + b2)
            error = target - o

            dO = o*(1-o)*error
            dW2 = dO*h
            db2 = dO
            
            # simplified hidden grad
            dH = dO * W2
            if act == relu:
                dH *= (h > 0)
            else:
                dH *= (1 - h*h)

            W2 += lr*dW2; b2 += lr*db2
            W  += lr*np.outer(xi, dH); b += lr*dH

    preds = [sigmoid(act(xi@W+b)@W2+b2) for xi in X]
    acc = sum((np.array(preds)>0.5).astype(int)==y)/len(y)
    return acc

print("TANH Accuracy:", train(tanh))
print("ReLU Accuracy:", train(relu))

# Implement FFN with custom weight initialization techniques and analyse their impact.
# init_compare.py
import numpy as np

X = np.array([[30,40],[50,60],[20,25],[70,75]])
y = np.array([0,1,0,1])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def one_epoch(W1):
    # fixed forward test
    x = np.array([50,60])
    h = sigmoid(x @ W1)
    return h.mean()

# Initializations
init_zeros = np.zeros((2,3))
init_small  = np.random.randn(2,3) * 0.01
init_xavier = np.random.randn(2,3) * np.sqrt(1/2)
init_he     = np.random.randn(2,3) * np.sqrt(2/2)

print("Zero Init:", one_epoch(init_zeros))
print("Small Init:", one_epoch(init_small))
print("Xavier Init:", one_epoch(init_xavier))
print("He Init:", one_epoch(init_he))

# Implement FFN that produces both classification and regression outputs for marks.
# ffn_multi_output.py
import numpy as np

# X = 2 subject marks
X = np.array([[40,50],[60,70],[30,20],[80,85]])
y_class = np.array([0,1,0,1])           # binary classification
y_reg   = np.array([45,65,25,82])       # regression target

def sigmoid(x): return 1/(1+np.exp(-x))

# 2 → 3 → (1 classification + 1 regression)
W1 = np.random.randn(2,3)
b1 = np.zeros(3)

Wc = np.random.randn(3)   # classification head
bc = 0.0

Wr = np.random.randn(3)   # regression head
br = 0.0

lr = 0.001

for _ in range(400):
    for xi, yc, yr in zip(X, y_class, y_reg):
        h = sigmoid(xi@W1 + b1)

        # two outputs
        oc = sigmoid(h @ Wc + bc)
        orr = h @ Wr + br

        # errors
        ec = yc - oc
        er = yr - orr

        # grads
        dOc = oc*(1-oc)*ec
        dWc = dOc*h
        dbc = dOc

        dOr = er
        dWr = dOr*h
        dbr = dOr

        # hidden layer receives BOTH gradients
        dH = dOc*Wc + dOr*Wr
        dH *= h*(1-h)  # sigmoid derivative

        Wc += lr*dWc; bc += lr*dbc
        Wr += lr*dWr; br += lr*dbr
        W1 += lr*np.outer(xi, dH); b1 += lr*dH

# Predictions
hfull = sigmoid(X@W1 + b1)
class_out = (sigmoid(hfull@Wc + bc) > 0.5).astype(int)
reg_out   = hfull@Wr + br

print("Class predictions:", class_out)
print("Reg predictions:", reg_out)


# Write a Python program to implement Hebbian learning for storing binary patterns.

# 15_hebbian_auto.py
import numpy as np

# binary patterns (+1, -1)
patterns = [
    np.array([1, -1, 1]),
    np.array([-1, -1, 1])
]

# weight matrix
W = np.zeros((3,3))

# Hebbian learning: W = Σ x xᵀ
for p in patterns:
    W += np.outer(p, p)

# No self-connections
np.fill_diagonal(W, 0)

print("Learned Weight Matrix:\n", W)

# Test recall
test = np.array([1, -1, 1])
out = np.sign(test @ W)
print("Recall Output:", out)


# 16. Implement Hebb rule with noisy inputs and test its noise tolerance.

# 16_hebb_noise.py
import numpy as np
import random

p = np.array([1, -1, 1, -1])  # stored pattern

# Learn weights
W = np.outer(p, p)
np.fill_diagonal(W, 0)

# add noise (flip 1–2 bits randomly)
test = p.copy()
flip_positions = random.sample(range(4), 2)
for i in flip_positions:
    test[i] *= -1

print("Noisy Input:", test)

# recall
out = np.sign(test @ W)
print("Recalled Output:", out)


# Implement Hebb learning for hetero-associative memory (pattern A → pattern B).

# 17_hebb_hetero.py
import numpy as np

# A → B mapping patterns
A = np.array([[1,-1, 1],
              [-1,1,-1]])

B = np.array([[ 1, 1],
              [-1,1]])

# Hebb rule: W = Σ (Bᵀ A)
W = B.T @ A

print("Weight Matrix:\n", W)

# test recall: give A, get B
for i in range(len(A)):
    out = np.sign(W @ A[i])
    print(f"A{i} →", out)


# Demonstrate stability of weights when multiple patterns are trained sequentially using Hebb rule.
# 18_hebb_stability.py
import numpy as np

patterns = [
    np.array([1,-1,1]),
    np.array([-1,1,-1]),
    np.array([1,1,-1])
]

W = np.zeros((3,3))

# train patterns one-by-one and print weight matrix after each
for idx, p in enumerate(patterns):
    W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    print(f"After learning pattern {idx+1}:\n{W}\n")

# test stability (final recall)
for p in patterns:
    print("Recall:", np.sign(p @ W))


# Implement a 3-layer MLP from scratch to classify semester marks.
# 19_mlp_3layer.py
import numpy as np

# tiny dataset: [sub1, sub2] -> pass(1)/fail(0)
X = np.array([[30,40],[45,55],[70,80],[50,45]])
y = np.array([[0],[1],[1],[0]])

# network sizes
D, H, O = 2, 4, 1
np.random.seed(0)
W1 = np.random.randn(D, H) * 0.1
b1 = np.zeros((1,H))
W2 = np.random.randn(H, O) * 0.1
b2 = np.zeros((1,O))

def sigmoid(x): return 1/(1+np.exp(-x))

lr = 0.01
for epoch in range(1000):
    # forward
    H_act = sigmoid(X.dot(W1) + b1)
    O_act = sigmoid(H_act.dot(W2) + b2)
    # loss gradient (MSE simplified)
    error = y - O_act
    dO = error * O_act * (1 - O_act)
    dW2 = H_act.T.dot(dO)
    db2 = dO.sum(axis=0, keepdims=True)
    dH = dO.dot(W2.T) * H_act * (1 - H_act)
    dW1 = X.T.dot(dH)
    db1 = dH.sum(axis=0, keepdims=True)
    # update
    W2 += lr * dW2
    b2 += lr * db2
    W1 += lr * dW1
    b1 += lr * db1

pred = (O_act > 0.5).astype(int)
print("Final predictions:", pred.flatten(), "True:", y.flatten())


# Add two hidden layers and implement manual forward + backward propagation.

# 20_mlp_two_hidden.py
import numpy as np

X = np.array([[30,40],[45,55],[70,80],[50,45]])
y = np.array([[0],[1],[1],[0]])

D, H1, H2, O = 2, 5, 3, 1
np.random.seed(1)
W1 = np.random.randn(D, H1)*0.1; b1 = np.zeros((1,H1))
W2 = np.random.randn(H1, H2)*0.1; b2 = np.zeros((1,H2))
W3 = np.random.randn(H2, O)*0.1; b3 = np.zeros((1,O))

def sigmoid(x): return 1/(1+np.exp(-x))

lr = 0.01
for epoch in range(1500):
    # forward
    z1 = X.dot(W1) + b1; a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2; a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3; a3 = sigmoid(z3)
    # backprop
    err = y - a3
    d3 = err * a3*(1-a3)
    dW3 = a2.T.dot(d3); db3 = d3.sum(axis=0, keepdims=True)
    d2 = d3.dot(W3.T) * a2*(1-a2)
    dW2 = a1.T.dot(d2); db2 = d2.sum(axis=0, keepdims=True)
    d1 = d2.dot(W2.T) * a1*(1-a1)
    dW1 = X.T.dot(d1); db1 = d1.sum(axis=0, keepdims=True)
    # update
    W3 += lr*dW3; b3 += lr*db3
    W2 += lr*dW2; b2 += lr*db2
    W1 += lr*dW1; b1 += lr*db1

print("Pred:", (a3>0.5).astype(int).flatten(), "True:", y.flatten())


# Implement MLP using momentum-based weight updates.
# 21_mlp_momentum.py
import numpy as np

# small network 2->4->1
X = np.array([[30,40],[45,55],[70,80],[50,45]])
y = np.array([[0],[1],[1],[0]])

D,H,O = 2,4,1
np.random.seed(2)
W1 = np.random.randn(D,H)*0.1; b1 = np.zeros((1,H))
W2 = np.random.randn(H,O)*0.1; b2 = np.zeros((1,O))

def sigmoid(x): return 1/(1+np.exp(-x))

lr = 0.01; momentum = 0.9
vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

for epoch in range(800):
    a1 = sigmoid(X.dot(W1)+b1)
    a2 = sigmoid(a1.dot(W2)+b2)
    err = y - a2
    d2 = err * a2*(1-a2)
    dW2 = a1.T.dot(d2); db2 = d2.sum(axis=0,keepdims=True)
    d1 = d2.dot(W2.T) * a1*(1-a1)
    dW1 = X.T.dot(d1); db1 = d1.sum(axis=0,keepdims=True)
    # velocity updates
    vW2 = momentum*vW2 + lr*dW2; vb2 = momentum*vb2 + lr*db2
    vW1 = momentum*vW1 + lr*dW1; vb1 = momentum*vb1 + lr*db1
    W2 += vW2; b2 += vb2
    W1 += vW1; b1 += vb1

print("Pred:", (a2>0.5).astype(int).flatten(), "True:", y.flatten())


# Modify the MLP to perform multi-class grade prediction.

# 22_mlp_multiclass.py
import numpy as np

# classes: 0=Poor,1=Average,2=Good
X = np.array([[30,35],[45,50],[60,65],[80,85],[50,55]])
y = np.array([0,1,1,2,1])
# one-hot
Y = np.eye(3)[y]

D, H, C = 2, 6, 3
np.random.seed(3)
W1 = np.random.randn(D,H)*0.1; b1 = np.zeros((1,H))
W2 = np.random.randn(H,C)*0.1; b2 = np.zeros((1,C))

def relu(x): return np.maximum(0,x)
def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

lr = 0.01
for epoch in range(1000):
    h = relu(X.dot(W1)+b1)
    out = softmax(h.dot(W2)+b2)
    err = Y - out
    dW2 = h.T.dot(err); db2 = err.sum(axis=0,keepdims=True)
    dh = err.dot(W2.T)
    dh[h<=0] = 0
    dW1 = X.T.dot(dh); db1 = dh.sum(axis=0,keepdims=True)
    W2 += lr*dW2; b2 += lr*db2
    W1 += lr*dW1; b1 += lr*db1

pred = np.argmax(out, axis=1)
print("Pred:", pred, "True:", y, "Acc:", (pred==y).mean())



# 23. Implement dropout manually inside the MLP and test training stability.

# 23_mlp_dropout.py
import numpy as np

X = np.array([[30,40],[45,55],[70,80],[50,45]])
y = np.array([[0],[1],[1],[0]])
D,H,O = 2,6,1
np.random.seed(4)
W1 = np.random.randn(D,H)*0.1; b1 = np.zeros((1,H))
W2 = np.random.randn(H,O)*0.1; b2 = np.zeros((1,O))

def sigmoid(x): return 1/(1+np.exp(-x))
lr = 0.01; dropout_p = 0.4

for epoch in range(600):
    # forward with dropout on hidden during training
    z1 = X.dot(W1)+b1; a1 = sigmoid(z1)
    mask = (np.random.rand(*a1.shape) > dropout_p) / (1.0 - dropout_p)
    a1d = a1 * mask
    z2 = a1d.dot(W2)+b2; a2 = sigmoid(z2)
    # backprop
    err = y - a2
    d2 = err * a2*(1-a2)
    dW2 = a1d.T.dot(d2); db2 = d2.sum(axis=0,keepdims=True)
    d1 = d2.dot(W2.T) * a1*(1-a1) * mask
    dW1 = X.T.dot(d1); db1 = d1.sum(axis=0,keepdims=True)
    W2 += lr*dW2; b2 += lr*db2
    W1 += lr*dW1; b1 += lr*db1

# evaluate without dropout (scale expected)
a1_eval = sigmoid(X.dot(W1)+b1)
a2_eval = sigmoid(a1_eval.dot(W2)+b2)
print("Acc with dropout training:", ((a2_eval>0.5).astype(int)==y).mean())


# Compare MLP accuracy with different activation functions (ReLU vs Sigmoid).

# 24_relu_vs_sigmoid.py
import numpy as np

X = np.array([[30,40],[45,55],[70,80],[50,45]])
y = np.array([[0],[1],[1],[0]])

def train(act):
    D,H,O=2,6,1
    np.random.seed(5)
    W1 = np.random.randn(D,H)*0.1; b1 = np.zeros((1,H))
    W2 = np.random.randn(H,O)*0.1; b2 = np.zeros((1,O))
    def sigmoid(x): return 1/(1+np.exp(-x))
    lr = 0.01
    for _ in range(800):
        if act=="relu":
            a1 = np.maximum(0, X.dot(W1)+b1)
            a2 = sigmoid(a1.dot(W2)+b2)
        else:
            a1 = sigmoid(X.dot(W1)+b1)
            a2 = sigmoid(a1.dot(W2)+b2)
        err = y - a2
        d2 = err * a2*(1-a2)
        dW2 = a1.T.dot(d2); db2 = d2.sum(axis=0,keepdims=True)
        d1 = d2.dot(W2.T)
        if act=="relu":
            d1 *= (a1>0)
        else:
            d1 *= a1*(1-a1)
        dW1 = X.T.dot(d1); db1 = d1.sum(axis=0,keepdims=True)
        W2 += lr*dW2; b2 += lr*db2; W1 += lr*dW1; b1 += lr*db1
    pred = (a2>0.5).astype(int)
    return (pred==y).mean()

print("ReLU Acc:", train("relu"))
print("Sigmoid Acc:", train("sigmoid"))


# Implement MLP with early stopping and plot training/validation loss.
# 25_mlp_early_stopping.py
import numpy as np
import matplotlib.pyplot as plt

# simple split train/val
X = np.array([[30,40],[45,55],[70,80],[50,45],[55,60],[35,30]])
y = np.array([[0],[1],[1],[0],[1],[0]])
Xtr, ytr = X[:4], y[:4]
Xval, yval = X[4:], y[4:]

D,H,O=2,6,1
np.random.seed(6)
W1 = np.random.randn(D,H)*0.1; b1 = np.zeros((1,H))
W2 = np.random.randn(H,O)*0.1; b2 = np.zeros((1,O))

def sigmoid(x): return 1/(1+np.exp(-x))
lr=0.01
best_val_loss = 1e9; patience = 30; wait=0
train_losses=[]; val_losses=[]

for epoch in range(2000):
    # train epoch
    a1 = sigmoid(Xtr.dot(W1)+b1)
    a2 = sigmoid(a1.dot(W2)+b2)
    tr_loss = np.mean((ytr - a2)**2)
    # update
    err = ytr - a2
    d2 = err * a2*(1-a2)
    dW2 = a1.T.dot(d2); db2 = d2.sum(axis=0,keepdims=True)
    d1 = d2.dot(W2.T) * a1*(1-a1)
    dW1 = Xtr.T.dot(d1); db1 = d1.sum(axis=0,keepdims=True)
    W2 += lr*dW2; b2 += lr*db2; W1 += lr*dW1; b1 += lr*db1

    # val loss
    va1 = sigmoid(Xval.dot(W1)+b1); va2 = sigmoid(va1.dot(W2)+b2)
    val_loss = np.mean((yval - va2)**2)
    train_losses.append(tr_loss); val_losses.append(val_loss)

    # early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss; best_weights = (W1.copy(),b1.copy(),W2.copy(),b2.copy()); wait=0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping at epoch", epoch)
            W1,b1,W2,b2 = best_weights
            break

plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.show()


# Write a Python program to train MLP using backpropagation for XOR problem.

# 27_mlp_xor.py
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(7)
W1 = np.random.randn(2,2); b1 = np.zeros((1,2))
W2 = np.random.randn(2,1); b2 = np.zeros((1,1))
def sigmoid(x): return 1/(1+np.exp(-x))

lr=0.1
for epoch in range(10000):
    a1 = sigmoid(X.dot(W1)+b1)
    a2 = sigmoid(a1.dot(W2)+b2)
    err = y - a2
    d2 = err * a2*(1-a2)
    dW2 = a1.T.dot(d2); db2 = d2.sum(axis=0)
    d1 = d2.dot(W2.T) * a1*(1-a1)
    dW1 = X.T.dot(d1); db1 = d1.sum(axis=0)
    W2 += lr*dW2; b2 += lr*db2; W1 += lr*dW1; b1 += lr*db1

print("Pred:", np.round(a2,3).flatten(), "True:", y.flatten())


# 28. Implement backprop with dynamic learning rate and test for faster
convergence.
# 28_dynamic_lr.py
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
np.random.seed(8)
W1 = np.random.randn(2,2)*0.5; b1 = np.zeros((1,2))
W2 = np.random.randn(2,1)*0.5; b2 = np.zeros((1,1))
def sigmoid(x): return 1/(1+np.exp(-x))

initial_lr = 0.5
for epoch in range(1,5001):
    lr = initial_lr / (1 + 0.001*epoch)  # decay
    a1 = sigmoid(X.dot(W1)+b1)
    a2 = sigmoid(a1.dot(W2)+b2)
    err = y - a2
    d2 = err * a2*(1-a2)
    W2 += lr * a1.T.dot(d2); b2 += lr * d2.sum(axis=0)
    d1 = d2.dot(W2.T) * a1*(1-a1)
    W1 += lr * X.T.dot(d1); b1 += lr * d1.sum(axis=0)

print("Final preds:", np.round(a2,3).flatten())

# Implement ANN (MLP) for regression (predicting total marks) with backprop.

# 29_mlp_regression.py
import numpy as np

# X = five subject marks, y = total marks
X = np.array([[40,42,38,50,45],[60,62,65,58,59],[30,28,25,35,32]])
y = X.sum(axis=1, keepdims=True).astype(float)

np.random.seed(9)
D,H,O = 5,8,1
W1 = np.random.randn(D,H)*0.1; b1 = np.zeros((1,H))
W2 = np.random.randn(H,O)*0.1; b2 = np.zeros((1,O))
def relu(x): return np.maximum(0,x)

lr = 0.001
for epoch in range(2000):
    h = relu(X.dot(W1)+b1)
    out = h.dot(W2)+b2  # linear output for regression
    err = y - out
    dW2 = h.T.dot(err)* (2/len(X))
    db2 = err.sum(axis=0) * (2/len(X))
    dh = err.dot(W2.T)
    dh[h<=0] = 0
    dW1 = X.T.dot(dh)*(2/len(X)); db1 = dh.sum(axis=0)*(2/len(X))
    W2 += lr*dW2; b2 += lr*db2; W1 += lr*dW1; b1 += lr*db1

print("Pred totals:", out.flatten(), "True:", y.flatten())


# Modify backprop to include L2 regularization and observe changes.

# 30_mlp_l2.py
import numpy as np

X = np.array([[30,40],[45,55],[70,80],[50,45]])
y = np.array([[0],[1],[1],[0]])
D,H,O = 2,6,1
np.random.seed(10)
W1 = np.random.randn(D,H)*0.1; b1 = np.zeros((1,H))
W2 = np.random.randn(H,O)*0.1; b2 = np.zeros((1,O))

def sigmoid(x): return 1/(1+np.exp(-x))
lr = 0.01; l2 = 0.01
for epoch in range(800):
    a1 = sigmoid(X.dot(W1)+b1)
    a2 = sigmoid(a1.dot(W2)+b2)
    err = y - a2
    d2 = err * a2*(1-a2)
    dW2 = a1.T.dot(d2) - l2 * W2
    db2 = d2.sum(axis=0)
    d1 = d2.dot(W2.T) * a1*(1-a1)
    dW1 = X.T.dot(d1) - l2 * W1
    db1 = d1.sum(axis=0)
    W2 += lr*dW2; b2 += lr*db2; W1 += lr*dW1; b1 += lr*db1

print("Acc:", ((a2>0.5).astype(int)==y).mean())

# Test performance of backprop ANN on random noisy marks dataset.

# 31_mlp_noisy_test.py
import numpy as np
np.random.seed(11)
# generate 200 samples, 5 subjects, class=1 if mean>=50 else 0, add noise
X = np.random.normal(loc=50, scale=15, size=(200,5))
y = (X.mean(axis=1) >= 50).astype(int).reshape(-1,1)
# add label noise flips
flip = np.random.choice(200, size=10, replace=False)
y[flip] = 1 - y[flip]

# simple network 5->8->1
D,H,O = 5,8,1
W1 = np.random.randn(D,H)*0.1; b1=np.zeros((1,H))
W2 = np.random.randn(H,O)*0.1; b2=np.zeros((1,O))
def sigmoid(x): return 1/(1+np.exp(-x))
lr=0.01
for epoch in range(300):
    a1 = sigmoid(X.dot(W1)+b1)
    a2 = sigmoid(a1.dot(W2)+b2)
    err = y - a2
    d2 = err * a2*(1-a2)
    W2 += lr*a1.T.dot(d2); b2 += lr*d2.sum(axis=0)
    d1 = d2.dot(W2.T) * a1*(1-a1)
    W1 += lr*X.T.dot(d1); b1 += lr*d1.sum(axis=0)

print("Train acc noisy dataset:", ((a2>0.5).astype(int)==y).mean())


# Provide visualization of gradients layer-wise for each epoch in backprop training.

# 32_grad_viz.py
import numpy as np
import matplotlib.pyplot as plt

# tiny dataset
X = np.array([[30,40],[45,55],[70,80],[50,45]])
y = np.array([[0],[1],[1],[0]])
D,H,O = 2,6,1
np.random.seed(12)
W1 = np.random.randn(D,H)*0.1; b1=np.zeros((1,H))
W2 = np.random.randn(H,O)*0.1; b2=np.zeros((1,O))
def sigmoid(x): return 1/(1+np.exp(-x))

lr = 0.01
epochs = 200
grad_hist_W1 = []
grad_hist_W2 = []

for epoch in range(epochs):
    a1 = sigmoid(X.dot(W1)+b1)
    a2 = sigmoid(a1.dot(W2)+b2)
    err = y - a2
    d2 = err * a2*(1-a2)
    dW2 = a1.T.dot(d2)
    d1 = d2.dot(W2.T) * a1*(1-a1)
    dW1 = X.T.dot(d1)
    # save gradient norms per layer
    grad_hist_W1.append(np.linalg.norm(dW1))
    grad_hist_W2.append(np.linalg.norm(dW2))
    # update
    W2 += lr*dW2; b2 += lr*d2.sum(axis=0)
    W1 += lr*dW1; b1 += lr*d1.sum(axis=0)

plt.plot(grad_hist_W1, label='||grad W1||')
plt.plot(grad_hist_W2, label='||grad W2||')
plt.xlabel('epoch'); plt.ylabel('gradient norm'); plt.legend(); plt.show()

# Write Python code to implement a CNN (conv + pooling + FC) without using inbuilt frameworks.
# 33_simple_cnn.py
import numpy as np

def conv2d(inp, kernel, stride=1, pad=0):
    # inp: HxW, kernel: kh x kw
    if pad:
        inp = np.pad(inp, pad, mode='constant')
    H, W = inp.shape
    kh, kw = kernel.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = inp[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i,j] = np.sum(patch * kernel)
    return out

def relu(x): return np.maximum(0, x)

def max_pool(mat, size=2, stride=2):
    H, W = mat.shape
    out = np.zeros(((H//stride), (W//stride)))
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            patch = mat[i:i+size, j:j+size]
            out[i//stride, j//stride] = np.max(patch)
    return out

# demo
img = np.random.randn(28,28)
kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])  # simple edge filter
fmap = relu(conv2d(img, kernel, pad=1))
pooled = max_pool(fmap, 2, 2)
vector = pooled.flatten()
# simple FC
Wfc = np.random.randn(vector.size, 10)
out = vector @ Wfc
print("Output logits shape:", out.shape)   # (10,)


# Implement manual convolution operation using nested loops and test on images.

# 34_manual_conv.py
import numpy as np

def manual_conv2d(inp, kernel):
    H, W = inp.shape; kh, kw = kernel.shape
    out_h = H - kh + 1; out_w = W - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            s = 0.0
            for u in range(kh):
                for v in range(kw):
                    s += inp[i+u, j+v] * kernel[u, v]
            out[i,j] = s
    return out

img = np.array([[1,2,3,0],
                [0,1,2,1],
                [3,1,0,2],
                [2,2,1,0]], dtype=float)
kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=float)
out = manual_conv2d(img, kernel)
print("Input:\n", img)
print("Kernel:\n", kernel)
print("Conv output:\n", out)


# Build a CNN for handwritten digit recognition (MNIST-subset).

# 35_mnist_cnn_simplified.py
import numpy as np

# dummy "MNIST-subset" (replace with real data: X shape (N,28,28), y shape (N,))
N = 200
X_train = np.random.randn(N,28,28)
y_train = np.random.randint(0,10,size=(N,))

# simple layer set up (single-channel)
def conv_single(img, kernel):  # same as earlier conv2d
    return conv2d(img, kernel, pad=1)

# random kernels
k1 = np.random.randn(3,3)
k2 = np.random.randn(3,3)
Wfc = np.random.randn((28//4)*(28//4), 10) * 0.01  # rough fc size after 2x2 pooling twice

def forward(img):
    f1 = np.maximum(0, conv2d(img, k1, pad=1))
    p1 = max_pool(f1, 2, 2)
    f2 = np.maximum(0, conv2d(p1, k2, pad=1))
    p2 = max_pool(f2, 2, 2)
    vec = p2.flatten()
    logits = vec @ Wfc
    return logits

# tiny "training" just runs forward and prints sample logits
for i in range(3):
    logits = forward(X_train[i])
    print("Sample", i, "logits shape:", logits.shape)


# Implement max pooling and average pooling manually and compare outputs.

# 36_pool_compare.py
import numpy as np

def max_pool(mat, size=2, stride=2):
    H,W = mat.shape
    out = np.zeros((H//stride, W//stride))
    for i in range(0,H,stride):
        for j in range(0,W,stride):
            out[i//stride, j//stride] = np.max(mat[i:i+size, j:j+size])
    return out

def avg_pool(mat, size=2, stride=2):
    H,W = mat.shape
    out = np.zeros((H//stride, W//stride))
    for i in range(0,H,stride):
        for j in range(0,W,stride):
            out[i//stride, j//stride] = np.mean(mat[i:i+size, j:j+size])
    return out

mat = np.arange(16).reshape(4,4)
print("Input:\n", mat)
print("MaxPool:\n", max_pool(mat,2,2))
print("AvgPool:\n", avg_pool(mat,2,2))


# Visualize feature maps generated after each convolution layer.

# 37_visualize_feature_maps.py
import numpy as np
import matplotlib.pyplot as plt

# simple grayscale image
img = np.random.randn(28,28)
# make 6 random 3x3 kernels
kernels = [np.random.randn(3,3) for _ in range(6)]

feature_maps = [np.maximum(0, conv2d(img, k, pad=1)) for k in kernels]

plt.figure(figsize=(10,4))
for i, fmap in enumerate(feature_maps):
    plt.subplot(1,6,i+1)
    plt.imshow(fmap, cmap='viridis')
    plt.axis('off')
    plt.title(f'FM{i+1}')
plt.tight_layout(); plt.show()


# Write Python code to build a simplified AlexNet architecture.

# 38_simplified_alexnet.py
import numpy as np

# simplified conv layer builder (C_out, kernel, input_channels)
def conv_params(cin, cout, k):
    return cin * cout * (k*k) + cout  # weights + biases

# hypothetical simplified AlexNet (single-channel for demo)
layers = [
    ('conv', (1, 64, 11)),   # in_ch, out_ch, k
    ('pool', None),
    ('conv', (64, 192, 5)),
    ('pool', None),
    ('conv', (192, 384, 3)),
    ('conv', (384, 256, 3)),
    ('conv', (256, 256, 3)),
    ('pool', None),
    ('fc', (256*6*6, 4096)),
    ('fc', (4096, 4096)),
    ('fc', (4096, 1000))
]

def count_params(layers):
    total = 0
    for t,p in layers:
        if t=='conv':
            cin, cout, k = p
            total += conv_params(cin, cout, k)
        elif t=='fc':
            in_dim, out_dim = p
            total += in_dim * out_dim + out_dim
    return total

print("Simplified AlexNet param count (approx):", count_params(layers))


# Implement a simplified VGG-16 network and compare its depth with AlexNet.

# 39_simplified_vgg16.py
# VGG-16: conv blocks -> fully connected. We'll use channel sizes roughly as original.
vgg_layers = [
    ('conv', (3,64,3)), ('conv',(64,64,3)), ('pool',None),
    ('conv',(64,128,3)), ('conv',(128,128,3)), ('pool',None),
    ('conv',(128,256,3)), ('conv',(256,256,3)), ('conv',(256,256,3)), ('pool',None),
    ('conv',(256,512,3)), ('conv',(512,512,3)), ('conv',(512,512,3)), ('pool',None),
    ('conv',(512,512,3)), ('conv',(512,512,3)), ('conv',(512,512,3)), ('pool',None),
    ('fc',(512*7*7,4096)), ('fc',(4096,4096)), ('fc',(4096,1000))
]

def count_layers(layers):
    conv_count = sum(1 for t,_ in layers if t=='conv')
    fc_count = sum(1 for t,_ in layers if t=='fc')
    return conv_count + fc_count

print("VGG simplified param count (approx):", count_params(vgg_layers))
print("VGG depth (conv+fc):", count_layers(vgg_layers))
# compare with previous AlexNet layers (from snippet 38)
alex_depth = count_layers(layers)
print("AlexNet depth (simplified):", alex_depth, "VGG deeper by", count_layers(vgg_layers)-alex_depth)


# Implement a mini-version of GoogLeNet with at least one Inception module.
# 40_minigooglenet.py
import numpy as np

def conv_sim(inp_chan, out_chan, k): 
    # return number of params for this conv
    return inp_chan * out_chan * (k*k) + out_chan

# one Inception module simulation: branches: 1x1, 1x1->3x3, 1x1->5x5, 3x3 pool->1x1
def inception_params(in_ch, b1, b3r, b3, b5r, b5, pool_proj):
    p = 0
    p += conv_sim(in_ch, b1, 1)
    p += conv_sim(in_ch, b3r, 1) + conv_sim(b3r, b3, 3)
    p += conv_sim(in_ch, b5r, 1) + conv_sim(b5r, b5, 5)
    p += conv_sim(in_ch, pool_proj, 1)
    return p

print("Inception params example:", inception_params(192, 64, 96, 128, 16, 32, 32))


# Compare number of parameters of AlexNet, VGG and GoogLeNet manually.
# 41_param_compare.py
# reuse counts from earlier (approx)
alex_params = count_params(layers)   # from 38
vgg_params = count_params(vgg_layers)  # from 39
goog_params = 0
# approximate GoogLeNet by few inception modules:
goog_params += conv_params(3,64,7)  # initial conv
goog_params += inception_params(64, 64, 96, 128, 16, 32, 32) * 3  # three inception repeats
goog_params += (1024*1000 + 1000)  # final fc approx
print("AlexNet (approx):", alex_params)
print("VGG-16 (approx):", vgg_params)
print("GoogLeNet (approx):", goog_params)


# Simulate forward pass of an Inception module and print outputs from each branch.

# 42_inception_forward_sim.py
import numpy as np

def conv_forward_sim(x, out_ch, k):
    # x shape (C,H,W). We'll simulate conv by returning shape only
    C,H,W = x.shape
    out_h = H - k + 1  # no padding for simplicity
    return np.zeros((out_ch, out_h, out_h))

def maxpool_forward_sim(x, pool=3, stride=1):
    C,H,W = x.shape
    out_h = (H - pool)//stride + 1
    return np.zeros((C, out_h, out_h))

# input feature map (192 channels, 28x28)
x = np.zeros((192,28,28))

# branches
b1 = conv_forward_sim(x, 64, 1)
b2_1 = conv_forward_sim(x, 96, 1); b2 = conv_forward_sim(b2_1, 128, 3)
b3_1 = conv_forward_sim(x, 16, 1); b3 = conv_forward_sim(b3_1, 32, 5)
bp = maxpool_forward_sim(x, pool=3, stride=1); bp_proj = conv_forward_sim(bp, 32, 1)

print("Branch1 shape:", b1.shape)
print("Branch2 shape:", b2.shape)
print("Branch3 shape:", b3.shape)
print("Pool->proj shape:", bp_proj.shape)
# concatenated output shape (channels summed)
total_channels = b1.shape[0] + b2.shape[0] + b3.shape[0] + bp_proj.shape[0]
print("Concatenated output shape:", (total_channels, b1.shape[1], b1.shape[2]))
