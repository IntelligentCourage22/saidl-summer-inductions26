
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

class WikiTextDataset(Dataset):
    def __init__(self, split, seq_len, tokenizer_name="gpt2"):
        self.seq_len = seq_len
        
        # load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # load dataset
        print(f"Loading WikiText-2 ({split})...")
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # join all text and tokenize in one shot (most efficient way)
        all_text = "\n\n".join(
            [t for t in raw["text"] if t.strip() != ""]
        )
        
        print(f"Tokenizing...")
        tokens = self.tokenizer.encode(all_text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        # how many full chunks of seq_len+1 do we have
        # +1 because input is tokens[:-1] and target is tokens[1:]
        total = len(self.tokens)
        self.n_chunks = (total - 1) // seq_len
        
        print(f"  Total tokens : {total:,}")
        print(f"  Chunks of {seq_len}: {self.n_chunks:,}")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # input: seq_len tokens, target: same shifted by 1
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


def get_dataloaders(seq_len, batch_size, num_workers=2):
    train_ds = WikiTextDataset("train", seq_len)
    val_ds   = WikiTextDataset("validation", seq_len)
    test_ds  = WikiTextDataset("test", seq_len)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# quick sanity check
if __name__ == "__main__":
    train_loader, val_loader, _ = get_dataloaders(
        seq_len=1024, batch_size=4
    )
    x, y = next(iter(train_loader))
    print(f"Input shape : {x.shape}")   # [4, 1024]
    print(f"Target shape: {y.shape}")   # [4, 1024]
    print("Dataset OK!")
