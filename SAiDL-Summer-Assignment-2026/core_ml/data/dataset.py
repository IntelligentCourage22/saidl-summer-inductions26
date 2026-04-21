
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

class WikiTextDataset(Dataset):
    def __init__(self, split, seq_len, tokenizer_name="gpt2"):
        self.seq_len = seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        all_text = "\n\n".join([t for t in raw["text"] if t.strip() != ""])
        tokens = self.tokenizer.encode(all_text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        total = len(self.tokens)
        self.n_chunks = (total - 1) // seq_len
        print(f"  Total tokens : {total:,}")
        print(f"  Chunks of {seq_len}: {self.n_chunks:,}")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y

def get_dataloaders(seq_len, batch_size, num_workers=2):
    train_ds = WikiTextDataset("train",      seq_len)
    val_ds   = WikiTextDataset("validation", seq_len)
    test_ds  = WikiTextDataset("test",       seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
