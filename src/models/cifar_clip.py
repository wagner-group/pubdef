"""Zero-shot classification with CLIP."""

from __future__ import annotations

import logging

import clip
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.types import BatchImages, BatchLogits

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATES = {
    "full": [
        "a bad photo of a {}.",
        "a photo of many {}.",
        "a sculpture of a {}.",
        "a photo of the hard to see {}.",
        "a low resolution photo of the {}.",
        "a rendering of a {}.",
        "graffiti of a {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a tattoo of a {}.",
        "the embroidered {}.",
        "a photo of a hard to see {}.",
        "a bright photo of a {}.",
        "a photo of a clean {}.",
        "a photo of a dirty {}.",
        "a dark photo of the {}.",
        "a drawing of a {}.",
        "a photo of my {}.",
        "the plastic {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a painting of the {}.",
        "a painting of a {}.",
        "a pixelated photo of the {}.",
        "a sculpture of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a plastic {}.",
        "a photo of the dirty {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a rendering of the {}.",
        "a {} in a video game.",
        "a photo of one {}.",
        "a doodle of a {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "the origami {}.",
        "the {} in a video game.",
        "a sketch of a {}.",
        "a doodle of the {}.",
        "a origami {}.",
        "a low resolution photo of a {}.",
        "the toy {}.",
        "a rendition of the {}.",
        "a photo of the clean {}.",
        "a photo of a large {}.",
        "a rendition of a {}.",
        "a photo of a nice {}.",
        "a photo of a weird {}.",
        "a blurry photo of a {}.",
        "a cartoon {}.",
        "art of a {}.",
        "a sketch of the {}.",
        "a embroidered {}.",
        "a pixelated photo of a {}.",
        "itap of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a plushie {}.",
        "a photo of the nice {}.",
        "a photo of the small {}.",
        "a photo of the weird {}.",
        "the cartoon {}.",
        "art of the {}.",
        "a drawing of the {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "the plushie {}.",
        "a dark photo of a {}.",
        "itap of a {}.",
        "graffiti of the {}.",
        "a toy {}.",
        "itap of my {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "a tattoo of the {}.",
    ],
    # Templates from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/datasets/en_zeroshot_classification_templates.json
    "laion": [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a bad photo of a {}.",
        "a good photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}.",
        "a photo of the {}.",
        "a blurry photo of the {}.",
        "a black and white photo of the {}.",
        "a low contrast photo of the {}.",
        "a high contrast photo of the {}.",
        "a bad photo of the {}.",
        "a good photo of the {}.",
        "a photo of the small {}.",
        "a photo of the big {}.",
    ],
}


_CLASSNAMES = {
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "cifar100": [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "cra",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ],
}

_AVAILABLE_MODELS = {
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B32",
    "ViT-B16",
    "ViT-L14",
    "ViT-L14@336px",
}


class CLIPClassifier(nn.Module):
    """Zero-shot classification with CLIP."""

    def __init__(
        self,
        num_classes: int = 10,
        clip_model: str = "RN50",
        prompt_template: str = "full",
        **kwrags,
    ) -> None:
        """Initialize a CLIPClassifier.

        Args:
            num_classes: Number of classes expected. Defaults to 10.
            clip_model: Backbone architecture of CLIP. Defaults to "RN50".
            prompt_template: Template for the prompt. Defaults to "full".

        Raises:
            ValueError: Invalid clip_model.
        """
        _ = kwrags  # Unused
        super().__init__()
        dataset = "cifar10" if num_classes == 10 else "cifar100"
        self._classnames = _CLASSNAMES[dataset]
        assert len(self._classnames) == num_classes, (
            f"Number of classes does not match dataset. {dataset} has "
            f"{len(self._classnames)} classes, but {num_classes} were given."
        )

        if clip_model not in _AVAILABLE_MODELS:
            raise ValueError(
                f"Model {clip_model} is not available. Available models are "
                f"{list(_AVAILABLE_MODELS)}!"
            )
        # Add slash to ViT models (it messes with the path otherwise)
        clip_model = clip_model.replace("ViT-B", "ViT-B/")
        clip_model = clip_model.replace("ViT-L", "ViT-L/")
        self._wrapped_model, _ = clip.load(clip_model)

        if prompt_template not in _PROMPT_TEMPLATES:
            raise ValueError(
                f"Prompt template {prompt_template} is not available. "
                f"Available templates are {list(_PROMPT_TEMPLATES.keys())}!"
            )
        self._prompt_template = prompt_template

        zeroshot_weights = self._classname_embed()
        self.register_buffer("zeroshot_weights", zeroshot_weights)
        logger.info(
            "CLIP model setup | backbone: %s, dataset: %s, num_classes: %d, "
            "prompt_template: %s",
            clip_model,
            dataset,
            num_classes,
            prompt_template,
        )

    @torch.no_grad()
    def _classname_embed(self):
        zeroshot_weights = []
        for classname in self._classnames:
            # Format all prompts with class names
            texts = [
                template.format(classname)
                for template in _PROMPT_TEMPLATES[self._prompt_template]
            ]
            # Tokenize
            texts = clip.tokenize(texts)
            # Embed with text encoder
            class_embeddings = self._wrapped_model.encode_text(texts.cuda())
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        return torch.stack(zeroshot_weights, dim=0)

    def forward(self, images: BatchImages) -> BatchLogits:
        """Forward pass of the model."""
        # Upscale images to 224x224 in differentiable manner
        images = F.interpolate(
            images, size=224, mode="bicubic", align_corners=False
        )
        image_features = self._wrapped_model.encode_image(images)
        norm = image_features.norm(dim=-1, keepdim=True)
        image_features = image_features / norm
        logits = torch.einsum(
            "bd, cd -> bc", image_features, self.zeroshot_weights
        )
        return logits
