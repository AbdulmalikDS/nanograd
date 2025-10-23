from nn import MLP
from loss import MSELoss
from optim import SGD, Adam, RMSprop

# Create a simple dataset
X = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
y = [1.0, -1.0, -1.0, 1.0]  # Target values

# Create model
model = MLP(3, [4, 4, 1])
loss_fn = MSELoss()

# Try different optimizers
print("=" * 50)
print("Training with SGD (with momentum)")
print("=" * 50)

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):
  # Forward pass
  y_pred = [model(x) for x in X]
  loss = loss_fn(y_pred, y)
  
  # Backward pass
  optimizer.zero_grad()
  loss.backward()
  
  # Update weights
  optimizer.step()
  
  if epoch % 10 == 0:
    print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")

print("\n" + "=" * 50)
print("Training with Adam")
print("=" * 50)

# Reset model
model = MLP(3, [4, 4, 1])
optimizer = Adam(model.parameters(), lr=0.01)

for epoch in range(100):
  y_pred = [model(x) for x in X]
  loss = loss_fn(y_pred, y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if epoch % 10 == 0:
    print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")

print("\n" + "=" * 50)
print("Training with RMSprop")
print("=" * 50)

# Reset model
model = MLP(3, [4, 4, 1])
optimizer = RMSprop(model.parameters(), lr=0.01)

for epoch in range(100):
  y_pred = [model(x) for x in X]
  loss = loss_fn(y_pred, y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if epoch % 10 == 0:
    print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")

print("\nFinal predictions:")
for i, x in enumerate(X):
  pred = model(x)
  print(f"Input: {x} | Predicted: {pred.data:.4f} | Target: {y[i]}")

