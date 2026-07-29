import sys

sys.path.append(".")

import numpy as np
from enums import SplitRunType
from utils import config_utils, gpu_utils, caption_utils
from data.schemas import DataItemSchema, PytorchModelOutputSchema
import constants

from bertopic import BERTopic
from bertopic.representation import TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
import torch


config = config_utils.load_config()


TOPIC_MODEL_PATH = "./topic_modelling/medic_topic_model/"


class NeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super(NeuralNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, device=device, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.model(x)


class CustomClassifierBERTopic:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.float64
        self.model: torch.nn.Module | None = None
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = None

    def fit(self, X, y, epochs=100, batch_size=32):
        X = torch.from_numpy(X).to(device=self.device, dtype=self.dtype)
        y = torch.from_numpy(y).to(device=self.device, dtype=torch.long)

        if self.model is None:
            input_size = X.shape[1]
            hidden_size = input_size
            output_size = len(constants.LABELS_MEDIC_DISASTER_TYPES)
            self.model = NeuralNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                device=self.device,
                dtype=self.dtype,
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        try:
            X = torch.from_numpy(X).to(dtype=self.dtype)
        except:
            X = torch.tensor(X).to(dtype=self.dtype)
        X = X.to(self.device)
        if self.model is None:
            input_size = X.shape[1]
            hidden_size = input_size
            output_size = len(constants.LABELS_MEDIC_DISASTER_TYPES)
            self.model = NeuralNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                device=self.device,
                dtype=self.dtype,
            )
        outputs = self.model(X)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy().tolist()


def train_bert_topic(docs, y):
    empty_dimensionality_model = BaseDimensionalityReduction()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    topic_model = BERTopic(
        embedding_model="distilroberta-base-msmarco-v1",
        umap_model=empty_dimensionality_model,
        hdbscan_model=CustomClassifierBERTopic(),
        ctfidf_model=ctfidf_model,
        top_n_words=15,
    )
    topic_model.fit_transform(docs, y=y)
    topic_model.save(
        TOPIC_MODEL_PATH,
        serialization="pytorch",
        save_ctfidf=True,
        save_embedding_model=empty_dimensionality_model,
    )
    print()
    for topic in range(len(constants.LABELS_MEDIC_DISASTER_TYPES)):
        topics = topic_model.get_topic(topic, full=False)
        print(topics)
    print()
    return topic_model


def get_topic_model(docs, y):
    if config.get("train_bert_topic"):
        print("----- TRAININING TOPIC MODEL -----")
        return train_bert_topic(docs, y)
    else:
        print("----- USING TRAINED TOPIC MODEL -----")
        return BERTopic.load(TOPIC_MODEL_PATH)


if __name__ == "__main__":
    from data.datasets import MedicDisasterDataset

    dataset = MedicDisasterDataset(split_run_type=SplitRunType.TRAIN)

    num_data_items = dataset.get_data_length()
    # num_data_items = 100

    train_df = dataset._get_dataframe_by_split_run_type()[:num_data_items]
    disaster_type_labels = np.array(
        train_df["disaster_types"]
        .map(constants.LABEL_NAME_TO_LABEL_VALUE_MEDIC_DISASTER_TYPES)
        .tolist()
    )

    texts = [dataset.get_data_item(i).metadata["caption"] for i in range(num_data_items)]
    texts = [caption_utils.clean_text(t) for t in texts]

    topic_model = get_topic_model(docs=texts, y=disaster_type_labels)
