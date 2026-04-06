from torch.utils.data import Dataset, DataLoader
import numpy as np


def nbeta_from_dataset(dataset: Dataset) -> float:
    m = len(dataset)
    return m / np.log(m)


def nbeta_from_loader(loader: DataLoader) -> float:
    return nbeta_from_dataset(loader.dataset)


def nbeta_from_effective_size(x: int) -> float:
    return x / np.log(x)
