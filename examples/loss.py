from engine import Value
from loss import MSELoss, MAELoss

# Create some predictions and targets
predictions = [Value(0.8), Value(0.3), Value(0.9)]
targets = [1.0, 0.0, 1.0]

print("Predictions:", [p.data for p in predictions])
print("Targets:    ", targets)
print()

# MSE Loss
mse = MSELoss()
loss_mse = mse(predictions, targets)
print(f"MSE Loss: {loss_mse.data:.6f}")

# Calculate gradients
loss_mse.backward()
print("Gradients (MSE):", [p.grad for p in predictions])

# Reset gradients
for p in predictions:
  p.grad = 0.0

# MAE Loss
mae = MAELoss()
loss_mae = mae(predictions, targets)
print(f"\nMAE Loss: {loss_mae.data:.6f}")

loss_mae.backward()
print("Gradients (MAE):", [p.grad for p in predictions])

