class EarlyStopping:
  def __init__(self, patience:int=7, min_delta:float=1e-7, verbose:bool=True, metric_name:str='Accuracy', mode:str='max'):
    self.patience = patience
    self.min_delta = min_delta
    self.verbose = verbose
    self.metric_name = metric_name
    self.mode = mode

    if mode == 'max':
      self.best_value = -float('inf')
      self.monitor_op = lambda current, best: current > best + self.min_delta
    else:
      self.best_value = float('inf')
      self.monitor_op = lambda current, best: current < best - self.min_delta

    self.counter = 0
    self.early_stop = False
    self.best_model_state = None

  def __call__(self, model, metric_value:float):
    if self.monitor_op(metric_value, self.best_value):
      self.best_value = metric_value
      self.counter = 0
      self.best_model_state = model.state_dict()
      if self.verbose:
        print(f'Validation {self.metric_name} improved to {metric_value:.4f}. Saving best model state...')
    else:
      self.counter += 1
      if self.verbose:
        print(f'No improvement in validation {self.metric_name}. Counter: {self.counter}/{self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True
        if self.verbose:
          print(f'Early stopping triggered after {self.patience} epochs without improvement')
