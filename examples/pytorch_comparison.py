import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Value

# Create values
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813, label='b')

# Forward pass
n = x1*w1 + x2*w2 + b
o = n.tanh()

# Backward pass
o.backward()
print(f"Output: {o.data}")
print("Gradients:")
print(f"x2: {x2.grad}")
print(f"w2: {w2.grad}")
print(f"x1: {x1.grad}")
print(f"w1: {w1.grad}")


import torch
    
# Create tensors with gradient tracking
x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.8813]).double(); b.requires_grad = True
    
# Forward pass
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)
    
# Backward pass
o.backward()

print("\n")
print("PyTorch")
print(f"Output: {o.data}")
print("Gradients:")
print(f"x2: {x2.grad.item()}")
print(f"w2: {w2.grad.item()}")
print(f"x1: {x1.grad.item()}")
print(f"w1: {w1.grad.item()}")
    
