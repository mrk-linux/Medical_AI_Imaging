import torch
from torch.utils.data import Dataset as TorchDataset


class PyTorchDataset(TorchDataset):
    """PyTorch dataset for prepared medical images."""

    def __init__(
        self,
        samples: list[tuple[object, int]],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self,index: int,) -> tuple[torch.Tensor, torch.Tensor]:

        image, label = self.samples[index]

        image_tensor = torch.from_numpy(image).unsqueeze(0)

        label_tensor = torch.tensor(label,dtype=torch.long,)

        return image_tensor, label_tensor