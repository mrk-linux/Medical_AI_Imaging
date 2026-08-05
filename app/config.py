from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    APP_NAME: str = "Medical AI Imaging"
    VERSION: str = "0.1.0"