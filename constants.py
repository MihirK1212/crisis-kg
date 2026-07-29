FIELD_IMAGE_TENSORS = "image_tensors"
FIELD_TOKENIZED_TEXT_INPUTS = "tokenized_text_inputs"
FIELD_LABEL = "label"
FIELD_METADATA = "metadata"
FIELD_SEED = "seed"
FIELD_TRAIN_SPLIT_RATIO = "train_split_ratio"
FIELD_MAX_LEN_TEXT = "max_len_text"
FIELD_TRAIN_BATCH_SIZE = "train_batch_size"
FIELD_VALIDATION_BATCH_SIZE = "validation_batch_size"
FIELD_TEST_BATCH_SIZE = "test_batch_size"
FIELD_PRED_LOGITS = "pred_logits"
FIELD_PYTROCH_MODEL_INTERFACE = "pytorch_model_interface"
FIELD_CLASSICAL_MODEL_INTERFACE = "classical_model_interface"
FIELD_APPLY_GRADIENT_CLIPPING = "apply_gradient_clipping"
FIELD_USE_LR_SCHEDULER = "use_lr_scheduler"
FIELD_DATASET = "dataset"
FIELD_MODEL_INTERFACE = "model_interface"
FIELD_RUNNER_INTERFACE = "runner_interface"
FIELD_MODEL_TO_USE = "model_to_use"
FIELD_DATASET_TO_USE = "dataset_to_use"
FIELD_EMBEDDING_DIM = "embedding_dim"
FIELD_IMAGE_EMBEDDING_DIM = "image_embedding_dim"
FIELD_TEXT_EMBEDDING_DIM = "text_embedding_dim"
FIELD_GLOBAL_EMBEDDING = "global_embedding"
FIELD_WORD_LEVEL_EMBEDDINGS = "word_level_embeddings"
FIELD_TEXT_EMBEDDINGS = "text_embeddings"
FIELD_IMAGE_EMBEDDINGS = "image_embeddings"
FIELD_TEXT = "text"
FIELD_USE_DUMMY_RANK_FOR_PARALLEL_ENABLED_MODEL = (
    "use_dummy_rank_for_parallel_enabled_model"
)
FIELD_RUN_PARALLEL = "run_parallel"
FIELD_NUM_EPOCHS = "num_epochs"
FIELD_READ_WRITE_GPU_RANK = "read_write_gpu_rank"
FIELD_RGB_PIXELS_TENSOR = "rgb_pixels_tensor"
FIELD_DISASTER_TYPE = "disaster_type"
FIELD_HUMANITARIAN_CATEGORY = "humanitarian_category"
FIELD_IMAGE_ID = "image_id"
FIELD_NUM_SELECTED_TOPICS = "num_selected_topics"
FIELD_INFORMATIVENESS_CATEGORY = "informativeness_category"

PATH_CONFIG_FILE = "config.yaml"

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"

############################### MEDIC DISASTER DATASET ###############################
PATH_MEDIC_DISASTER_DATASET_BASE_PATH = "./data/datasets/dataset_files/medic-crisis-nlp"
PATH_MEDIC_DISASTER_DATASET_TSV_DIR = (
    PATH_MEDIC_DISASTER_DATASET_BASE_PATH + "/" + "tsvs"
)

LABEL_MEDIC_EARTHQUAKE = "earthquake"
LABEL_MEDIC_NOT_DISASTER = "not_disaster"
LABEL_MEDIC_FLOOD = "flood"
LABEL_MEDIC_LANDSLIDE = "landslide"
LABEL_MEDIC_FIRE = "fire"
LABEL_MEDIC_HURRICANE = "hurricane"
LABEL_MEDIC_OTHER_DISASTER = "other_disaster"

LABEL_MEDIC_INFRASTRUCTURE_AND_UTILITY_DAMAGE = "infrastructure_and_utility_damage"
LABEL_MEDIC_AFFECTED_INJURED_DEAD_PEOPLE = "affected_injured_or_dead_people"
LABEL_MEDIC_RECUE_VOLUNTEERING_DONATION_EFFORT = (
    "rescue_volunteering_or_donation_effort"
)
LABEL_MEDIC_NOT_HUMANITARIAN = "not_humanitarian"

LABELS_MEDIC_DISASTER_TYPES = [
    LABEL_MEDIC_EARTHQUAKE,
    LABEL_MEDIC_NOT_DISASTER,
    LABEL_MEDIC_FLOOD,
    LABEL_MEDIC_LANDSLIDE,
    LABEL_MEDIC_FIRE,
    LABEL_MEDIC_HURRICANE,
    LABEL_MEDIC_OTHER_DISASTER,
]

LABELS_MEDIC_HUMANITARIAN_CATEGORIES = [
    LABEL_MEDIC_INFRASTRUCTURE_AND_UTILITY_DAMAGE,
    LABEL_MEDIC_AFFECTED_INJURED_DEAD_PEOPLE,
    LABEL_MEDIC_RECUE_VOLUNTEERING_DONATION_EFFORT,
    LABEL_MEDIC_NOT_HUMANITARIAN,
]

LABEL_NAME_TO_LABEL_VALUE_MEDIC_DISASTER_TYPES = {
    LABEL_MEDIC_EARTHQUAKE: 0,
    LABEL_MEDIC_FLOOD: 1,
    LABEL_MEDIC_LANDSLIDE: 2,
    LABEL_MEDIC_FIRE: 3,
    LABEL_MEDIC_HURRICANE: 4,
    LABEL_MEDIC_OTHER_DISASTER: 5,
    LABEL_MEDIC_NOT_DISASTER: 6,
}

LABEL_NAME_TO_LABEL_VALUE_MEDIC_HUMANITARIAN_CATEGORY = {
    LABEL_MEDIC_INFRASTRUCTURE_AND_UTILITY_DAMAGE: 0,
    LABEL_MEDIC_AFFECTED_INJURED_DEAD_PEOPLE: 1,
    LABEL_MEDIC_RECUE_VOLUNTEERING_DONATION_EFFORT: 2,
    LABEL_MEDIC_NOT_HUMANITARIAN: 3,
}

NUM_LABELS_MEDIC = {}
####################################################################################

############################### MODEL_NAMES ###############################
MODEL_MULTIMODAL_CONCEPT_TOPIC_GRAPH = "multimodal_concept_topic_graph"
####################################################################################
