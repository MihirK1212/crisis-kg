from re import U
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from data.schemas import DataItemSchema, PytorchModelOutputSchema
import constants

import torch
import torch.nn as nn

from utils import config_utils, gpu_utils
from .blocks import RGATLayer, CAGM, AKF
from transformers import BertTokenizer, BertModel, ViTModel


class MultimodalConceptTopicGraphModel(nn.Module):
    def __init__(self, num_classes, category_name: str) -> None:
        super(MultimodalConceptTopicGraphModel, self).__init__()

        self.config = config_utils.load_config()

        assert (
            self.config[constants.FIELD_MODEL_TO_USE]
            == constants.MODEL_MULTIMODAL_CONCEPT_TOPIC_GRAPH
        )

        self.num_classes = num_classes
        self.category_name = category_name

        # Common feature dimension
        self.d_model = 768

        # Text encoder (BERT)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # Vision encoder (ViT)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # Relational GAT for knowledge propagation
        relations = ["cc", "ct"]
        self.rgat_layers = nn.ModuleList(
            [
                RGATLayer(self.d_model, self.d_model, relations=relations),
                RGATLayer(self.d_model, self.d_model, relations=relations),
            ]
        )
        self.rgat_activation = nn.LeakyReLU(0.2)

        # CAGM: Concept-Anchored Graph Modulation (latent reasoning stream)
        self.cagm = CAGM(self.d_model)

        # AKF: Adaptive Knowledge Fusion (multi-view consensus gating + gated cross-modal attention)
        self.akf = AKF(self.d_model)

        # Final classifier on fused representation
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    @classmethod
    def validate_inputs(
        cls,
        batch_size: int,
        rgb_pixels_tensor: torch.Tensor,
        batch_captions: list[str],
        topics: list[dict],
        batch_concept_triplets: list[list[tuple[str, str, str]]],
    ):
        config = config_utils.load_config()

        assert (
            rgb_pixels_tensor.shape[0] == batch_size
        ), "RGB pixels tensor batch size does not match batch size"

        assert (
            len(batch_captions) == batch_size
        ), "Caption batch size does not match batch size"
        for caption in batch_captions:
            assert isinstance(caption, str), "Caption is not a string"
            assert len(caption) > 0, "Caption is empty"

        assert len(topics) == config.get(
            constants.FIELD_NUM_SELECTED_TOPICS
        ), "Number of topics does not match number of selected topics"

        assert (
            len(batch_concept_triplets) == batch_size
        ), "Concept triplets batch size does not match batch size"
        for concept_triplets in batch_concept_triplets:
            for concept_triplet in concept_triplets:
                assert isinstance(
                    concept_triplet, tuple
                ), "Concept triplet is not a tuple"
                assert (
                    len(concept_triplet) == 3
                ), "Concept triplet does not have 3 elements"
                assert isinstance(concept_triplet[0], str), "Subject is not a string"
                assert isinstance(concept_triplet[1], str), "Predicate is not a string"
                assert isinstance(concept_triplet[2], str), "Object is not a string"

    def forward(self, data_item: DataItemSchema) -> PytorchModelOutputSchema:
        batch_size = data_item.image_tensors[
            self.category_name
        ].rgb_pixels_tensor.shape[0]

        rgb_pixels_tensor = data_item.image_tensors[
            self.category_name
        ].rgb_pixels_tensor

        batch_captions = data_item.metadata["caption"]

        topics = data_item.metadata["topics"][0].split(
            ";"
        )  # topics are same across all data items, since they are generated for the entire dataset

        batch_concept_triplets = data_item.metadata["caption_concept_triplets"]

        batch_concept_triplets = [
            concept_triplets.split(";") for concept_triplets in batch_concept_triplets
        ]
        batch_concept_triplets = [
            [concept_triplet.split(":") for concept_triplet in concept_triplets]
            for concept_triplets in batch_concept_triplets
        ]
        batch_concept_triplets = [
            [
                (concept_triplet[0], concept_triplet[1], concept_triplet[2])
                for concept_triplet in concept_triplets
            ]
            for concept_triplets in batch_concept_triplets
        ]

        MultimodalConceptTopicGraphModel.validate_inputs(
            batch_size,
            rgb_pixels_tensor,
            batch_captions,
            topics,
            batch_concept_triplets,
        )

        device = gpu_utils.get_device()

        # Helper to encode a list of texts with BERT -> [num_texts, d_model]
        def encode_texts(texts: List[str]) -> torch.Tensor:
            encoded = self.bert_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.get(constants.FIELD_MAX_LEN_TEXT, 128),
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = self.text_encoder(**encoded)
            # Mean pool over valid tokens
            last_hidden = outputs.last_hidden_state  # [B, L, d]
            attn_mask = encoded["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            summed = (last_hidden * attn_mask).sum(dim=1)
            counts = attn_mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            return mean_pooled  # [B, d]

        # Topics are dataset-wide; parse once
        meta_topics = data_item.metadata["topics"]
        if isinstance(meta_topics, list):
            topics_texts_full = meta_topics[0].split(";")
        else:
            topics_texts_full = meta_topics.split(";")

        logits_list: List[torch.Tensor] = []

        for b in range(batch_size):
            # 1) Unimodal features
            image = rgb_pixels_tensor[b].unsqueeze(0).to(device)  # [1, 3, 224, 224]
            vit_out = self.vit(pixel_values=image)
            FV = vit_out.last_hidden_state.squeeze(0)  # [N_v, d]

            caption_text = batch_captions[b]
            text_enc = self.bert_tokenizer(
                caption_text,
                padding=True,
                truncation=True,
                max_length=self.config.get(constants.FIELD_MAX_LEN_TEXT, 128),
                return_tensors="pt",
            )
            text_enc = {k: v.to(device) for k, v in text_enc.items()}
            text_out = self.text_encoder(**text_enc)
            FL_full = text_out.last_hidden_state.squeeze(0)  # [N_l, d]

            # 2) Build heterogeneous graph V = Vc ∪ Vt
            concept_triplets: List[Tuple[str, str, str]] = batch_concept_triplets[b]
            concepts: List[str] = []
            for s, p, o in concept_triplets:
                if s not in concepts:
                    concepts.append(s)
                if o not in concepts:
                    concepts.append(o)

            topic_texts: List[str] = topics_texts_full

            # Initial node embeddings using BERT text encoder
            concept_emb = (
                encode_texts(concepts)
                if len(concepts) > 0
                else torch.zeros((0, self.d_model), device=device)
            )
            topic_emb = (
                encode_texts(topic_texts)
                if len(topic_texts) > 0
                else torch.zeros((0, self.d_model), device=device)
            )
            node_feats = torch.cat([concept_emb, topic_emb], dim=0)  # [|V|, d]

            # Indices
            concept_to_idx: Dict[str, int] = {c: i for i, c in enumerate(concepts)}
            topic_offset = len(concepts)

            # Edges
            edges: Dict[str, List[Tuple[int, int]]] = {"cc": [], "ct": []}
            # Rcc: directed s->o
            for s, p, o in concept_triplets:
                if s in concept_to_idx and o in concept_to_idx:
                    edges["cc"].append((concept_to_idx[s], concept_to_idx[o]))

            # Rct: concept->topic if cosine similarity exceeds tau
            if len(concepts) > 0 and len(topic_texts) > 0:
                tau = 0.3
                c_norm = torch.nn.functional.normalize(concept_emb, dim=-1)
                t_norm = torch.nn.functional.normalize(topic_emb, dim=-1)
                sims = c_norm @ t_norm.transpose(0, 1)  # [C, T]
                c_idx, t_idx = torch.nonzero(sims > tau, as_tuple=True)
                for ci, ti in zip(c_idx.tolist(), t_idx.tolist()):
                    edges["ct"].append((ci, topic_offset + ti))

            # 3) RGAT propagation (explicit stream)
            Fk = node_feats
            for layer in self.rgat_layers:
                Fk = layer(Fk, edges)
                Fk = self.rgat_activation(Fk)

            if Fk.shape[0] == 0:
                Fk = torch.zeros((1, self.d_model), device=device)

            # Build concept prototype set from explicit stream: use only concept nodes portion if available
            num_concepts = len(concepts)
            if num_concepts > 0:
                P = Fk[:num_concepts]  # [C, d]
            else:
                P = torch.zeros((0, self.d_model), device=device)

            # Build concept-concept adjacency matrix for CALT bias
            if num_concepts > 0:
                adj_cc = torch.zeros((num_concepts, num_concepts), device=device)
                for s, p, o in concept_triplets:
                    if s in concept_to_idx and o in concept_to_idx:
                        si = concept_to_idx[s]
                        oi = concept_to_idx[o]
                        if si < num_concepts and oi < num_concepts:
                            adj_cc[si, oi] = 1.0
            else:
                adj_cc = torch.zeros((0, 0), device=device)

            # 4) CAGM: concept-anchored graph modulation (latent reasoning stream)
            Flatent_V, Flatent_L = self.cagm(FV=FV, FL=FL_full, P=P, adj_cc=adj_cc)

            # 5) AKF: adaptive knowledge fusion
            Ffused = self.akf(FV=FV, FL=FL_full, Fexp=Fk, Flatent_V=Flatent_V, Flatent_L=Flatent_L)

            # 6) Classification using fused [CLS] token (first token of FL sequence)
            if Ffused.shape[0] > 0:
                h_cls = Ffused[0]
            else:
                h_cls = torch.zeros((self.d_model,), device=device)
            logit = self.classifier(h_cls.unsqueeze(0))  # [1, num_classes]
            logits_list.append(logit)

        multimodal_concept_topic_graph_logits = torch.cat(logits_list, dim=0)

        return PytorchModelOutputSchema(
            **{
                constants.FIELD_PRED_LOGITS: {
                    self.category_name: multimodal_concept_topic_graph_logits,
                }
            }
        )

    def get_bert_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Return mean-pooled BERT embeddings for a list of texts.

        Args:
            texts: List of input strings to encode.

        Returns:
            Tensor of shape [len(texts), d_model] on the current device.
        """
        device = gpu_utils.get_device()
        self.text_encoder.eval()
        with torch.no_grad():
            encoded = self.bert_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.get(constants.FIELD_MAX_LEN_TEXT, 128),
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = self.text_encoder(**encoded)
            last_hidden = outputs.last_hidden_state  # [B, L, d]
            attn_mask = encoded["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            summed = (last_hidden * attn_mask).sum(dim=1)
            counts = attn_mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            return mean_pooled
