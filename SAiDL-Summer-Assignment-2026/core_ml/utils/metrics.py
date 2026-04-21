
import torch, time, math

class MetricsTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
    def update(self, loss, n_tokens):
        self.total_loss   += loss * n_tokens
        self.total_tokens += n_tokens
    def get_perplexity(self):
        return math.exp(min(self.total_loss / self.total_tokens, 100))
    def get_throughput(self):
        return self.total_tokens / (time.time() - self.start_time)
    def get_peak_memory_mb(self):
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    def get_summary(self):
        return {"perplexity": self.get_perplexity(), "throughput_tps": self.get_throughput(), "peak_memory_mb": self.get_peak_memory_mb()}

def compute_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)
