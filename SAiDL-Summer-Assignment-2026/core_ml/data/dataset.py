import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    def __init__(
        self,
        split,
        seq_len,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        tokenizer_name="gpt2",
    ):
        self.seq_len = seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        raw = load_dataset(dataset_name, dataset_config, split=split)
        all_text = "\n\n".join(text for text in raw["text"] if text.strip())
        tokens = self.tokenizer.encode(all_text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        total = len(self.tokens)
        self.n_chunks = (total - 1) // seq_len
        if self.n_chunks == 0:
            raise ValueError(
                f"Split {split} has {total} tokens, too few for seq_len={seq_len}."
            )

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


def get_dataloaders(
    seq_len,
    batch_size,
    num_workers=2,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer_name="gpt2",
):
    kwargs = {
        "seq_len": seq_len,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "tokenizer_name": tokenizer_name,
    }
    train_ds = WikiTextDataset("train", **kwargs)
    val_ds = WikiTextDataset("validation", **kwargs)
    test_ds = WikiTextDataset("test", **kwargs)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
