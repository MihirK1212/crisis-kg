import torch
from typing import List, Optional, Dict, Union, Any, Tuple
from pydantic import BaseModel, PositiveInt, Field

import constants


class ImageTensorsSchema(BaseModel):
    rgb_pixels_tensor: torch.Tensor = Field(alias=constants.FIELD_RGB_PIXELS_TENSOR)

    class Config:
        arbitrary_types_allowed = True


class TokenizedTextInputsSchema(BaseModel):
    ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


# these are dicts because there may be different inputs for different label classes
class DataItemSchema(BaseModel):
    image_tensors: Dict[str, ImageTensorsSchema] = Field(
        alias=constants.FIELD_IMAGE_TENSORS
    )
    tokenized_text_inputs: Dict[str, TokenizedTextInputsSchema] = Field(
        alias=constants.FIELD_TOKENIZED_TEXT_INPUTS
    )
    label: Dict[str, torch.Tensor] = Field(alias=constants.FIELD_LABEL)
    topics_tokenized_inputs: List[Dict[str, torch.Tensor]]
    metadata: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed = True


class PytorchModelOutputSchema(BaseModel):
    pred_logits: Dict[str, torch.Tensor] = Field(alias=constants.FIELD_PRED_LOGITS)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


class LogLineSchema(BaseModel):
    data: List[Tuple[str, Any]]
    new_line_in_between: bool
    blank_line_after: bool