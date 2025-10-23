class Optimizer:
  
  def __init__(self, parameters):
    self.parameters = parameters
  
  def zero_grad(self):
    for p in self.parameters:
      p.grad = 0.0
  
  def step(self):
    raise NotImplementedError


class SGD(Optimizer):
  
  def __init__(self, parameters, lr=0.01, momentum=0.0):
    super().__init__(parameters)
    self.lr = lr
    self.momentum = momentum
    self.velocity = [0.0 for _ in parameters]
  
  def step(self):
    for i, p in enumerate(self.parameters):
      self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
      p.data += self.velocity[i]


class Adam(Optimizer):

  def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    super().__init__(parameters)
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.t = 0
    self.m = [0.0 for _ in parameters]  # First moment
    self.v = [0.0 for _ in parameters]  # Second moment
  
  def step(self):
    self.t += 1
    for i, p in enumerate(self.parameters):
      # Update biased first moment estimate
      self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
      # Update biased second raw moment estimate
      self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
      
      # Compute bias-corrected first moment estimate
      m_hat = self.m[i] / (1 - self.beta1 ** self.t)
      # Compute bias-corrected second raw moment estimate
      v_hat = self.v[i] / (1 - self.beta2 ** self.t)
      
      # Update parameters
      p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)


class RMSprop(Optimizer):
  
  def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8):
    super().__init__(parameters)
    self.lr = lr
    self.alpha = alpha
    self.eps = eps
    self.v = [0.0 for _ in parameters]
  
  def step(self):
    for i, p in enumerate(self.parameters):
      # Update moving average of squared gradients
      self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (p.grad ** 2)
      # Update parameters
      p.data -= self.lr * p.grad / (self.v[i] ** 0.5 + self.eps)

