
import torch
import time
import math


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss   = 0.0
        self.total_tokens = 0
        self.start_time   = time.time()
        torch.cuda.reset_peak_memory_stats()

    def update(self, loss, n_tokens):
        self.total_loss   += loss * n_tokens
        self.total_tokens += n_tokens

    def get_perplexity(self):
        avg_loss = self.total_loss / self.total_tokens
        return math.exp(min(avg_loss, 100))

    def get_throughput(self):
        elapsed = time.time() - self.start_time
        return self.total_tokens / elapsed

    def get_peak_memory_mb(self):
        return torch.cuda.max_memory_allocated() / 1024 / 1024

    def get_summary(self):
        return {
            "perplexity"     : self.get_perplexity(),
            "throughput_tps" : self.get_throughput(),
            "peak_memory_mb" : self.get_peak_memory_mb(),
        }


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)
