
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch, wandb, math
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.dataset   import get_dataloaders
from models.model   import TransformerLM
from utils.metrics  import MetricsTracker, compute_grad_norm


def evaluate(model, val_loader, device):
    model.eval()
    tracker = MetricsTracker()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            if loss.dim() > 0: loss = loss.mean()
            tracker.update(loss.item(), x.numel())
    model.train()
    return {"val_loss": tracker.total_loss/tracker.total_tokens,
            "val_perplexity": tracker.get_perplexity(),
            "peak_memory_mb": tracker.get_peak_memory_mb()}


def train(cfg):
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    run_name = f"{cfg.model.attention_type}_{cfg.model.positional_encoding}_{cfg.model.block_type}_ctx{cfg.training.seq_len}"

    wandb.init(project=cfg.wandb.project, name=run_name,
               config=OmegaConf.to_container(cfg, resolve=True),
               tags=[cfg.model.attention_type, cfg.model.positional_encoding, f"ctx_{cfg.training.seq_len}"])

    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(cfg.training.seq_len, cfg.training.batch_size, cfg.data.num_workers)
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    model = TransformerLM(cfg).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9,0.95), weight_decay=0.1)
    total_steps = len(train_loader) * cfg.training.num_epochs
    scheduler = SequentialLR(optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.training.warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - cfg.training.warmup_steps, eta_min=1e-5)
        ], milestones=[cfg.training.warmup_steps])

    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    global_step, best_val_ppl = 0, float("inf")

    for epoch in range(cfg.training.num_epochs):
        tracker = MetricsTracker()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            if loss.dim() > 0: loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()
            tracker.update(loss.item(), x.numel())
            global_step += 1

            if global_step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                wandb.log({"train/loss": loss.item(), "train/perplexity": math.exp(min(loss.item(),100)),
                           "train/grad_norm": grad_norm, "train/lr": lr,
                           "train/throughput": tracker.get_throughput(),
                           "train/memory_mb": tracker.get_peak_memory_mb()}, step=global_step)
            if global_step % 200 == 0:
                print(f"Step {global_step:5d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | ppl {math.exp(min(loss.item(),100)):.1f}")

            if global_step % cfg.training.eval_every == 0:
                vm = evaluate(model, val_loader, device)
                print(f"\n>>> Step {global_step} | val_loss {vm['val_loss']:.4f} | val_ppl {vm['val_perplexity']:.2f} | mem {vm['peak_memory_mb']:.0f}MB")
                wandb.log({"val/loss": vm["val_loss"], "val/perplexity": vm["val_perplexity"], "val/memory_mb": vm["peak_memory_mb"]}, step=global_step)
                if vm["val_perplexity"] < best_val_ppl:
                    best_val_ppl = vm["val_perplexity"]
                    raw = model.module if hasattr(model, "module") else model
                    torch.save({"step": global_step, "model": raw.state_dict(), "val_ppl": best_val_ppl,
                                "cfg": OmegaConf.to_container(cfg)}, f"/kaggle/working/checkpoints/best_{run_name}.pt")
                    print(f"  ✓ Saved best (ppl={best_val_ppl:.2f})")

        s = tracker.get_summary()
        print(f"\nEpoch {epoch+1}/{cfg.training.num_epochs} | train_ppl {s['perplexity']:.2f} | {s['throughput_tps']:.0f} tok/s | {s['peak_memory_mb']:.0f}MB\n")
        wandb.log({"epoch/train_perplexity": s["perplexity"], "epoch/throughput": s["throughput_tps"], "epoch": epoch+1})

    wandb.finish()
    print(f"Done. Best val ppl: {best_val_ppl:.2f}")
    return best_val_ppl

if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "configs", "config.yaml"))
    train(cfg)
