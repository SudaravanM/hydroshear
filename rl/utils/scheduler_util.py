

class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008, max_lr=1e-2, initial_cond=True):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = max_lr
        self.kl_threshold = kl_threshold
        self.initial_cond = initial_cond

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if self.initial_cond:
            if kl_dist > (2.0 * self.kl_threshold):
                lr = max(current_lr / 1.5, self.min_lr)
            if kl_dist < (0.5 * self.kl_threshold):
                lr = min(current_lr * 1.5, self.max_lr)
        else:
            if kl_dist < self.kl_threshold:
                self.initial_cond = True
        return lr