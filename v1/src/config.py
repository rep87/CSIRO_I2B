from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathConfig:
    data_root: str = "/content/drive/MyDrive/csiro-biomass"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    train_dir: str = "train"
    test_dir: str = "test"
    output_root: str = "./outputs"
    run_name: Optional[str] = None

    def resolve_train_csv(self) -> str:
        return f"{self.data_root}/{self.train_csv}"

    def resolve_test_csv(self) -> str:
        return f"{self.data_root}/{self.test_csv}"

    def resolve_train_dir(self) -> str:
        return f"{self.data_root}/{self.train_dir}"

    def resolve_test_dir(self) -> str:
        return f"{self.data_root}/{self.test_dir}"


@dataclass
class TrainConfig:
    backbone: str = "efficientnet_b2"
    image_size: int = 456
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 42
    folds: int = 5
    debug: bool = False
    accumulate_steps: int = 1
    amp: bool = True


@dataclass
class OptunaConfig:
    use_optuna: bool = False
    n_trials: int = 10
    timeout_minutes: int = 30
    storage: Optional[str] = None
    study_name: str = "csiro_optuna"


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    device: str = "cuda"

    def adjust_for_debug(self):
        if self.train.debug:
            self.train.epochs = 1
            self.train.batch_size = min(self.train.batch_size, 8)
            self.train.num_workers = 0
            self.train.accumulate_steps = 1
