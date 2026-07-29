import torch
import random
import json
from typing import List, Dict
import pandas as pd
from transformers import RobertaTokenizer
from PIL import Image, ImageFile
from torchvision import transforms


from data.dataset_interface import DatasetInterface
from data.schemas import DataItemSchema, ImageTensorsSchema, TokenizedTextInputsSchema
from utils import config_utils, data_utils, caption_utils
import constants
from enums import SplitRunType

from topic_modelling.medic import get_topic_model as get_medic_topic_model

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MedicDisasterDataset(DatasetInterface):
    def __init__(
        self,
        split_run_type: SplitRunType,
        tokenizer=caption_utils.get_default_tokenizer(),
    ) -> None:
        super().__init__(split_run_type)
        self._config = config_utils.load_config()

        assert self._config[constants.FIELD_DATASET_TO_USE] == "medic_disaster"

        self.dataframes_split = self._load_dataframes_split()
        self.dataframe = self._load_dataframe()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the image
                transforms.ToTensor(),
            ]
        )
        self.tokenizer = tokenizer
        self.analyse_dataframe()

        self.topics_tokenized_inputs = self._get_topics_tokenized_inputs()
        self.topics = MedicDisasterDataset.get_topics()

    def analyse_dataframe(self):
        print()
        print(self.split_run_type)
        print("size:", self.dataframe.shape)
        print(self.dataframe.head())
        print()

    def stratified_sample(self, df, sample_fraction=0.4):
        # Get rows for each label in class_a
        class_a_rows = [
            df[df["disaster_types"] == label].sample(n=1, random_state=42)
            for label in constants.LABELS_MEDIC_DISASTER_TYPES
            if label in df["disaster_types"].values
        ]
        class_a_rows_df = pd.concat(class_a_rows).drop_duplicates()

        # Get rows for each label in class_b
        class_b_rows = [
            df[df["humanitarian"] == label].sample(n=1, random_state=42)
            for label in constants.LABELS_MEDIC_HUMANITARIAN_CATEGORIES
            if label in df["humanitarian"].values
        ]
        class_b_rows_df = pd.concat(class_b_rows).drop_duplicates()

        # Combine rows from both classes
        label_rows_df = pd.concat([class_a_rows_df, class_b_rows_df]).drop_duplicates()

        # Determine how many more rows to sample
        required_rows = int(len(df) * sample_fraction)
        additional_rows = required_rows - len(label_rows_df)

        # Randomly sample additional rows excluding the already selected ones
        remaining_df = df.drop(label_rows_df.index)
        additional_rows_df = remaining_df.sample(
            n=max(additional_rows, 0), random_state=42
        )

        # Combine selected rows
        result_df = pd.concat([label_rows_df, additional_rows_df]).drop_duplicates()
        return result_df

    def _filter_dataframe_by_caption_existence(self, df: pd.DataFrame) -> pd.DataFrame:
        caption_exists_mask = [
            caption_utils.check_if_caption_triplet_exists(
                MedicDisasterDataset.get_image_id(df, i)
            )
            for i in range(len(df))
        ]
        return df[caption_exists_mask]

    def _load_dataframes_split(self) -> Dict[SplitRunType, pd.DataFrame]:
        train_df = pd.read_csv(
            f"{constants.PATH_MEDIC_DISASTER_DATASET_TSV_DIR}/MEDIC_train.tsv", sep="\t"
        )
        validation_df = pd.read_csv(
            f"{constants.PATH_MEDIC_DISASTER_DATASET_TSV_DIR}/MEDIC_dev.tsv", sep="\t"
        )
        test_df = pd.read_csv(
            f"{constants.PATH_MEDIC_DISASTER_DATASET_TSV_DIR}/MEDIC_test.tsv", sep="\t"
        )

        train_df = self._filter_dataframe_by_caption_existence(train_df)
        validation_df = self._filter_dataframe_by_caption_existence(validation_df)
        test_df = self._filter_dataframe_by_caption_existence(test_df)

        # train_df = self.stratified_sample(train_df)
        # validation_df = self.stratified_sample(validation_df)
        # test_df = self.stratified_sample(test_df)

        assert sorted(list(set(train_df["disaster_types"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_DISASTER_TYPES))
        )
        assert sorted(list(set(validation_df["disaster_types"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_DISASTER_TYPES))
        )
        assert sorted(list(set(test_df["disaster_types"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_DISASTER_TYPES))
        )

        assert sorted(list(set(train_df["humanitarian"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_HUMANITARIAN_CATEGORIES))
        )
        assert sorted(list(set(validation_df["humanitarian"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_HUMANITARIAN_CATEGORIES))
        )
        assert sorted(list(set(test_df["humanitarian"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_HUMANITARIAN_CATEGORIES))
        )

        dataframes_split = {
            SplitRunType.TRAIN: train_df,
            SplitRunType.VALIDATION: validation_df,
            SplitRunType.TEST: test_df,
        }
        return dataframes_split

    def _get_dataframe_by_split_run_type(self) -> pd.DataFrame:
        return self.dataframes_split[self.split_run_type]

    def _load_dataframe(self) -> pd.DataFrame:
        dataframe = self._get_dataframe_by_split_run_type()
        return dataframe

    def get_data_length(self) -> int:
        return len(self.dataframe)

    def get_data_item(self, idx=-1) -> DataItemSchema:
        relative_image_path = self.dataframe.iloc[idx]["image_path"]
        image_path = (
            f"{constants.PATH_MEDIC_DISASTER_DATASET_BASE_PATH}/{relative_image_path}"
        )
        image = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)

        image_id = MedicDisasterDataset.get_image_id(self.dataframe, idx)

        caption = ""
        caption_concept_triplets = []
        cot_triplets = {}
        try:
            if caption_utils.check_if_caption_exists(image_id):
                caption = caption_utils.get_caption(
                    image_id=image_id, image_file_path=image_path
                )
            else:
                caption = "THIS IS A TEST CAPTION. THE IMAGE SHOWS A DOG. THE DOG IS IN A FIELD. THE FIELD IS GREEN."

            caption_concept_triplets = caption_utils.get_concept_triplets(
                image_id=image_id, image_file_path=image_path
            )

            # CoT triplets are optional for the base multimodal_concept_topic_graph model
            try:
                cot_triplets = caption_utils.get_cot_triplets(
                    image_id=image_id, image_file_path=image_path
                )
            except Exception:
                cot_triplets = {}
        except Exception as e:
            caption = ""
            caption_concept_triplets = []
            cot_triplets = {}
            print(e)

        if not cot_triplets:
            cot_triplets = {}

        # num_triplets = random.randint(3, 10)
        # caption_concept_triplets = [
        #     (f"S{i}", f"P{i}", f"O{i}") for i in range(1, num_triplets + 1)
        # ]

        # cot_triplets = {
        #     "steps": [
        #         "Observe the scene for visible damage or hazards.",
        #         "Identify key entities present in the image.",
        #         "Determine relationships between the identified entities.",
        #         "Classify the overall crisis type depicted.",
        #     ],
        #     "entities": ["scene", "damage", "people"],
        #     "relations": [
        #         ["scene", "contains", "damage"],
        #         ["people", "affected by", "damage"],
        #         ["damage", "caused by", "disaster"],
        #     ],
        #     "crisis_type": "other_disaster",
        # }

        caption_text_tokenized_inputs = data_utils.parse_tokenizer_output(
            self.tokenizer(
                caption,
                None,
                add_special_tokens=True,
                max_length=self._config.get(constants.FIELD_MAX_LEN_TEXT),
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )
        )

        disaster_type_label_name = self.dataframe.iloc[idx]["disaster_types"]
        disaster_type_label = constants.LABEL_NAME_TO_LABEL_VALUE_MEDIC_DISASTER_TYPES[
            disaster_type_label_name
        ]

        return DataItemSchema(
            image_tensors={
                constants.FIELD_DISASTER_TYPE: ImageTensorsSchema(
                    **{constants.FIELD_RGB_PIXELS_TENSOR: image_tensor}
                )
            },
            tokenized_text_inputs={
                constants.FIELD_DISASTER_TYPE: caption_text_tokenized_inputs
            },
            topics_tokenized_inputs=self.topics_tokenized_inputs,
            label={
                constants.FIELD_DISASTER_TYPE: torch.tensor(
                    disaster_type_label, dtype=torch.long
                ),
            },
            metadata={
                constants.FIELD_IMAGE_ID: str(image_id),
                "image_path": relative_image_path,
                "absolute_image_path": image_path,
                "topics": ";".join([t["topic"] for t in self.topics]),
                "caption": caption,
                "caption_concept_triplets": ";".join(
                    [":".join(triplet) for triplet in caption_concept_triplets]
                ),
                "cot_triplets": json.dumps(cot_triplets) if cot_triplets else "{}",
            },
        )

    ###### TOPICS TOKENS #######

    def _get_topics_tokenized_inputs(self) -> List[TokenizedTextInputsSchema]:
        if self._config.get("train_bert_topic"):
            return []

        topics = MedicDisasterDataset.get_topics()

        topics_tokenized_inputs = [
            data_utils.parse_tokenizer_output(
                self.tokenizer(
                    topic["topic"],
                    None,
                    add_special_tokens=True,
                    max_length=3,
                    padding="max_length",
                    return_token_type_ids=True,
                )
            ).model_dump()
            for topic in topics
        ]
        return topics_tokenized_inputs

    @classmethod
    def get_image_id(cls, df: pd.DataFrame, idx: int) -> str:
        return (
            df.iloc[idx]["image_id"]
            + df.iloc[idx]["event_name"]
            + df.iloc[idx]["image_path"]
        )

    @classmethod
    def get_topics(cls) -> list[str]:
        config = config_utils.load_config()
        if config.get("train_bert_topic"):
            return [
                {"topic": f"Topic {i}", "score": 0.0}
                for i in range(config.get(constants.FIELD_NUM_SELECTED_TOPICS))
            ]
        topics = []
        topic_model = get_medic_topic_model(None, None)
        num_classes = len(constants.LABELS_MEDIC_DISASTER_TYPES)
        for topic in range(num_classes):
            curr_topics = topic_model.get_topic(topic, full=False)
            # sort by score
            curr_topics = sorted(curr_topics, key=lambda x: x[1], reverse=True)
            curr_topics = [{"topic": t[0], "score": t[1]} for t in curr_topics][
                : config.get(constants.FIELD_NUM_SELECTED_TOPICS) // num_classes
            ]
            topics.extend(curr_topics)
        assert len(topics) == config.get(constants.FIELD_NUM_SELECTED_TOPICS)
        return topics

        # return [
        #     {"topic": f"Topic {i}", "score": random.random()}
        #     for i in range(config.get(constants.FIELD_NUM_SELECTED_TOPICS))
        # ]

    #####################
