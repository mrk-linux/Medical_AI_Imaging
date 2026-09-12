from torch.utils.data import DataLoader

from app.input.pytorch_dataset import PyTorchDataset


class PyTorchDataLoader:
    """Create PyTorch DataLoader for medical image dataset."""

    def __init__(self,dataset: PyTorchDataset,batch_size: int = 32,shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def create(self) -> DataLoader:
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=self.shuffle)