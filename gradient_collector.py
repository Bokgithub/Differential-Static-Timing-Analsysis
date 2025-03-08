class GradientCollector:
    def __init__(self):
        self.gradients = {}
    
    def hook_factory(self, key):
        def hook(grad):
            if grad is not None:
                self.gradients[key] = grad.clone()
        return hook