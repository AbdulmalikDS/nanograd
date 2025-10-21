from nn import MLP
from engine import Value

# Training data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

# Create a 3-layer MLP
n = MLP(3, [4, 4, 1])

print(f"Number of parameters: {len(n.parameters())}")

# Training loop
for k in range(20):

  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum([(yout - Value(ygt))**2 for ygt, yout in zip(ys, ypred)], Value(0.0))

  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  # update
  for p in n.parameters():
    p.data += -0.05 * p.grad

  print(k, loss.data)

print("\nPredictions:")
print(ypred)

