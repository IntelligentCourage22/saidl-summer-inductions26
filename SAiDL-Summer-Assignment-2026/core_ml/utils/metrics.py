import math
import time

import torch


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def update(self, loss, n_tokens):
        self.total_loss += float(loss) * n_tokens
        self.total_tokens += n_tokens

    def get_average_loss(self):
        return self.total_loss / max(self.total_tokens, 1)

    def get_perplexity(self):
        return math.exp(min(self.get_average_loss(), 100))

    def get_throughput(self):
        elapsed = max(time.time() - self.start_time, 1e-9)
        return self.total_tokens / elapsed

    def get_peak_memory_mb(self):
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024 / 1024

    def get_summary(self):
        return {
            "loss": self.get_average_loss(),
            "perplexity": self.get_perplexity(),
            "throughput_tps": self.get_throughput(),
            "peak_memory_mb": self.get_peak_memory_mb(),
        }


def compute_grad_norm(model):
    total_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total_norm += parameter.grad.detach().data.norm(2).item() ** 2
    return math.sqrt(total_norm)
