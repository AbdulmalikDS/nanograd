from engine import Value

class Loss:
  
  def __call__(self, y_pred, y_true):
    raise NotImplementedError

class MSELoss(Loss):
  
  def __call__(self, y_pred, y_true):
    if not isinstance(y_pred, list):
      y_pred = [y_pred]
    if not isinstance(y_true, list):
      y_true = [y_true]
    
    losses = [(yp - yt)**2 for yp, yt in zip(y_pred, y_true)]
    total = losses[0]
    for loss in losses[1:]:
      total = total + loss
    return total * (1.0 / len(losses))


class BinaryCrossEntropyLoss(Loss):

  def __call__(self, y_pred, y_true):
    if not isinstance(y_pred, list):
      y_pred = [y_pred]
    if not isinstance(y_true, list):
      y_true = [y_true]
    
    eps = Value(1e-7)  # Small constant to avoid log(0)
    losses = []
    for yp, yt in zip(y_pred, y_true):
      yt = yt if isinstance(yt, Value) else Value(yt)
      # BCE = -[y*log(p) + (1-y)*log(1-p)]
      loss = -(yt * (yp + eps).log() + (Value(1) - yt) * (Value(1) - yp + eps).log())
      losses.append(loss)
    
    total = losses[0]
    for loss in losses[1:]:
      total = total + loss
    return total * (1.0 / len(losses))


class MAELoss(Loss):
  
  def __call__(self, y_pred, y_true):
    if not isinstance(y_pred, list):
      y_pred = [y_pred]
    if not isinstance(y_true, list):
      y_true = [y_true]
    
    losses = [(yp - yt).abs() for yp, yt in zip(y_pred, y_true)]
    total = losses[0]
    for loss in losses[1:]:
      total = total + loss
    return total * (1.0 / len(losses))

