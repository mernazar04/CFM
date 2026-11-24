
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PDE_Dataset(Dataset):
    def __init__(self, split="train"):
        ace = np.load(f"data/normalized/ace_{split}.npy")
        wg = np.load(f"data/normalized/wg_{split}.npy")

        # Объединяем два класса
        data = np.concatenate([ace, wg], axis=0)

        # Метки классов
        ace_labels = np.ones(len(ace), dtype=np.int64)
        wg_labels = np.full(len(wg), 2, dtype=np.int64)

        labels = np.concatenate([ace_labels, wg_labels])

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    train_dataset = PDE_Dataset("train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )

    x, labels = next(iter(train_loader))

    print("Batch shape:", x.shape)
    print("Labels:", labels)
    print("Dataset size:", len(train_dataset))