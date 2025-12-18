import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathConfig:
    data_root: str = "/content/drive/MyDrive/csiro-biomass"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    output_root: str = "./outputs"
    run_name: Optional[str] = None
    train_dir: Optional[str] = None
    test_dir: Optional[str] = None

    def resolve_train_csv(self) -> str:
        return os.path.join(self.data_root, self.train_csv)

    def resolve_test_csv(self) -> str:
        return os.path.join(self.data_root, self.test_csv)

    def resolve_train_dir(self) -> str:
        return self.data_root

    def resolve_test_dir(self) -> str:
        return self.data_root

    def resolve_image_root(self) -> str:
        """Base directory used to resolve image_path entries in the CSV files."""
        return self.data_root


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
    cv_split_strategy: str = "sequential"
    crop_bottom: float = 0.0
    use_clahe: bool = False
    scheduler: str = "plateau"  # plateau or cosine
    dropout: float = 0.0
    loss_beta: float = 1.0


@dataclass
class TuningFastDevConfig:
    enabled: bool = True
    epochs: int = 5
    batch_size_override: Optional[int] = None
    folds_subset: Optional[int] = 2


@dataclass
class TuningSearchSpace:
    lr: tuple = (1e-5, 3e-3)
    weight_decay: tuple = (1e-6, 1e-2)
    batch_size: tuple = (16, 32)  # categorical
    image_size: tuple = (384, 456)  # categorical
    dropout: tuple = (0.0, 0.3)
    loss_beta: tuple = (0.5, 2.0)
    scheduler: tuple = ("plateau", "cosine")  # categorical
    crop_bottom: tuple = (0.0, 0.05, 0.1)  # categorical
    use_clahe: tuple = (False, True)


@dataclass
class TuningConfig:
    enabled: bool = False
    n_trials: int = 20
    timeout_sec: Optional[int] = None
    direction: str = "maximize"
    study_name: Optional[str] = None
    storage: Optional[str] = None
    pruner: str = "median"  # median/none
    sampler: str = "tpe"  # tpe/random
    fast_dev: TuningFastDevConfig = field(default_factory=TuningFastDevConfig)
    search_space: TuningSearchSpace = field(default_factory=TuningSearchSpace)


@dataclass
class RuntimeConfig:
    use_optuna: bool = False
    use_fulltrain: bool = True


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    device: str = "cuda"

    def adjust_for_debug(self):
        if self.train.debug:
            self.train.epochs = 1
            self.train.batch_size = min(self.train.batch_size, 8)
            self.train.num_workers = 0
            self.train.accumulate_steps = 1

        if not 0.0 <= self.train.crop_bottom <= 0.3:
            raise ValueError("train.crop_bottom must be between 0.0 and 0.3")
