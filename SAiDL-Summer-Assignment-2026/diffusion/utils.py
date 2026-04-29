import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_config(path, overrides=None):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        key, raw_value = override.split("=", 1)
        set_nested(config, key, yaml.safe_load(raw_value))

    return config, to_namespace(config)


def set_nested(config, dotted_key, value):
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def maybe_init_wandb(config):
    if not config.get("wandb", {}).get("enabled", False):
        return None

    import wandb

    return wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        config=config,
    )


def torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for name, shadow_param in self.shadow.items():
            shadow_param.mul_(self.decay).add_(state[name].detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]

    def copy_to(self, model):
        state = model.state_dict()
        for name, shadow_param in self.shadow.items():
            state[name].copy_(shadow_param)
        model.load_state_dict(state)
