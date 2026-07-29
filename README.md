# Crisis-KG

Standalone repository for the **Multimodal Concept Topic Graph** model — a multimodal disaster-type classifier that combines BERT text encoding, ViT visual encoding, relational graph attention (RGAT) over concept–topic graphs, Concept-Anchored Graph Modulation (CAGM), and Adaptive Knowledge Fusion (AKF).

This repo contains only the machinery needed to **train, validate, and test** this model on the MEDIC crisis dataset.

## Structure

```
crisis-kg/
├── run.py                          # Train / val / test entry point
├── config.yaml                     # Hyperparameters
├── constants.py / enums.py
├── models/multimodal_concept_topic_graph/
│   ├── model.py                    # MultimodalConceptTopicGraphModel
│   └── blocks.py                   # RGATLayer, CAGM, AKF
├── runner_builders/                # Wires model + data + loss + optimizers
├── runner_interfaces/              # Epoch loop, metrics, logging
├── model_interfaces/               # Predict / fit / save wrappers
├── model_meta_components/          # Cross-entropy loss
├── data/                           # Schemas, MEDIC dataset, dataloaders
├── utils/                          # Config, GPU, metrics, caption helpers
├── topic_modelling/                # BERTopic topics for MEDIC
└── image_textual_metadata_generation/
    └── llava_*.py                  # Caption / concept-triplet generation (Ollama)
```

## Setup

```bash
cd crisis-kg
python -m venv myenv
# Windows: myenv\Scripts\activate
# Linux/macOS: source myenv/bin/activate
pip install -r requirements.txt
```

### Hugging Face models (downloaded automatically)

- `bert-base-uncased`
- `google/vit-base-patch16-224`
- `roberta-base` (dataset tokenizer)

### Optional: Ollama + LLaVA (for generating captions / triplets)

```bash
ollama serve
ollama pull llava
```

If captions and concept triplets are already cached as JSON under `image_textual_metadata_generation/`, Ollama is not required at train time.

## Data assets (not included)

Place or symlink the following before training:

| Asset | Expected path |
|-------|----------------|
| MEDIC images + TSVs | `data/datasets/dataset_files/medic-crisis-nlp/` |
| Train/dev/test TSVs | `.../medic-crisis-nlp/tsvs/MEDIC_{train,dev,test}.tsv` |
| Saved BERTopic model | `topic_modelling/medic_topic_model/` |
| Caption / triplet JSON caches | `image_textual_metadata_generation/{captions,triplets}/` |

Example symlink (adjust source path):

```bash
# From crisis-kg root
ln -s "/path/to/medic-crisis-nlp" data/datasets/dataset_files/medic-crisis-nlp
ln -s "/path/to/medic_topic_model" topic_modelling/medic_topic_model
```

To train topics from scratch, set `train_bert_topic: true` in `config.yaml` and run:

```bash
python -m topic_modelling.medic
```

## Configuration

Edit [`config.yaml`](config.yaml):

```yaml
model_to_use: multimodal_concept_topic_graph
dataset_to_use: medic_disaster
num_epochs: 10
train_batch_size: 4
num_selected_topics: 70
```

## Train / validate / test

Run from the **repository root** (so top-level imports resolve):

```bash
python run.py
```

This runs TRAIN → VALIDATION → TEST each epoch. Checkpoints are saved to:

```
models/multimodal_concept_topic_graph/saved_model/multimodal_concept_topic_graph_medic_disaster.pth
```

Logs are written under `logs/`.

## Model overview

Per sample:

1. **ViT** → visual tokens `FV`; **BERT** → caption tokens `FL`
2. Build a heterogeneous graph from concept triplets + dataset topics (`cc`, `ct` edges)
3. **RGAT** propagates knowledge on the graph (explicit stream)
4. **CAGM** performs concept-anchored latent reasoning
5. **AKF** fuses perceptual / explicit / latent views
6. Linear classifier on the fused text representation → 7 MEDIC disaster classes

## License / citation

Extracted from the Multimodal Concept Graph research codebase for paper reproduction.
# crisis-kg
