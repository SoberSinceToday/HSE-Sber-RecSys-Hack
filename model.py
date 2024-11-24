from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
import pandas as pd


class MyModel:

    def __init__(self, yaml_path: str):
        self.model = None
        self.TOP_K = 10
        self.EPOCHS = 50
        self.BATCH_SIZE = 1024
        self.SEED = DEFAULT_SEED

        self.hparams = prepare_hparams(yaml_path,
                                       n_layers=3,
                                       batch_size=self.BATCH_SIZE,
                                       epochs=self.EPOCHS,
                                       learning_rate=0.005,
                                       eval_epoch=5,
                                       top_k=self.TOP_K,
                                       )

    def fit(self, data: ImplicitCF):
        self.model = LightGCN(self.hparams, data, seed=self.SEED)
        with Timer() as train_time:
            self.model.fit()
        print(f"Took {train_time.interval} seconds for training.")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.model.recommend_k_items(data, top_k=self.TOP_K)
