from pathlib import Path
import torch
from models.cnn_model import CNNModel


class ModelSaver:
    """Save and load CNN model weights."""

    def save(self,model: CNNModel,path: str,) -> None:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True,exist_ok=True,)

        torch.save(model.state_dict(),model_path,)

    def load(self,model: CNNModel,path: str,) -> CNNModel:
        model.load_state_dict(torch.load(path,map_location="cpu",))

        return model