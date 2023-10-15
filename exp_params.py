"""Experiment parameters."""

from __future__ import annotations

RESULT_PATH = "./results/"
DATASETS: list[str] = ["cifar10", "cifar100", "imagenet"]
ATTACKS: list[str] = [
    "pgd",
    "mpgd",
    "pregradient",
    "di",
    "ti",
    "admix",
    "naa",
    "ni-si-ti-dim",
    "ni-admix-ti-dim",
    "auto-pgd-dlr",
    "auto-pgd-ce",
]
PUBLIC_MODELS: dict[str, dict[str, list[str]]] = {
    "cifar10": {
        "normal": [
            "cifar10_huggingface-vit-base",
            "cifar10_huggingface-beit",
            "cifar10_huggingface-convnext",
            "cifar10_chenyaofo-resnet20",
            "cifar10_chenyaofo-vgg11-bn",
            "cifar10_chenyaofo-mobilenetv2-x0-5",
            "cifar10_chenyaofo-shufflenetv2-x0-5",
            "cifar10_chenyaofo-repvgg-a0",
            "cifar10_huyvnphan-densenet121",
            "cifar10_huyvnphan-inception-v3",
            "cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw",
            "cifar10_clip-RN50-laion",
        ],
        "linf": [
            "cifar10_robustbench-wang23-wrn-70-16",
            "cifar10_robustbench-xu23-wrn-28-10",
            "cifar10_robustbench-debenedetti22-xcit-l12",
            "cifar10_robustbench-sehwag20-wrn-28-10",
        ],
        "l2": [
            "cifar10_robustbench-wang23-wrn-70-16-l2",
            "cifar10_robustbench-rebuffi21-wrn-70-16-l2",
            "cifar10_robustbench-augustin20-wrn-34-10-l2",
            "cifar10_robustbench-rony19-wrn-28-10-l2",
        ],
        "corruption": [
            "cifar10_robustbench-diffenderfer21-wrn-18-2",
            "cifar10_robustbench-kireev21-rn-18",
            "cifar10_robustbench-hendrycks20-resnext-29-32x4d",
            "cifar10_robustbench-modas21-rn-18",
        ],
    },
    "cifar100": {
        "normal": [
            "cifar100_chenyaofo-resnet20",
            "cifar100_chenyaofo-vgg11-bn",
            "cifar100_chenyaofo-mobilenetv2-x0-5",
            "cifar100_chenyaofo-shufflenetv2-x0-5",
            "cifar100_chenyaofo-repvgg-a0",
            "cifar100_huggingface-vit-base",
            "cifar100_huggingface-swin-tiny",
            "cifar100_densenet121_local",
            "cifar100_senet18_local",
            "cifar100_inception-v3_local",
            "cifar100_convmixer_local",
            "cifar100_clip-RN50-laion",
        ],
        "linf": [
            "cifar100_robustbench-wang23-wrn-70-16",
            "cifar100_robustbench-cui23-wrn-28-10",
            "cifar100_robustbench-bai23-mix",
            "cifar100_robustbench-debenedetti22-xcit",
            "cifar100_robustbench-jia22-wrn-34-20",
            "cifar100_robustbench-rade21-rn-18",
        ],
        "corruption": [
            "cifar100_robustbench-diffenderfer21-wrn-18-2",
            "cifar100_robustbench-modas21-rn-18",
            "cifar100_robustbench-hendrycks20-resnext-29-32x4d",
            "cifar100_robustbench-addepalli22-wrn-34-10",
            "cifar100_robustbench-gowal20-wrn-70-16",
            "cifar100_robustbench-diffenderfer21-bin-wrn-18-2",
        ],
    },
    "imagenet": {
        "normal": [
            "imagenet_huggingface-rn-50",
            "imagenet_huggingface-vit-b",
            "imagenet_huggingface-swin-t",
            "imagenet_huggingface-convnext-t",
            "imagenet_timm-efficientnet-b3",
            "imagenet_timm-inception-v3",
            "imagenet_timm-mnasnet-100",
            "imagenet_timm-mixer-b16",
            "imagenet_timm-rexnet-100",
            "imagenet_timm-hrnet-w18",
            "imagenet_timm-mobilenet-v3-l",
            "imagenet_timm-vgg-11",
        ],
        "linf": [
            "imagenet_robustbench-liu23-swin-b",
            "imagenet_robustbench-singh23-convnext-t",
            "imagenet_robustbench-singh23-vit-s",
            "imagenet_robustbench-debenedetti22-xcit-s12",
            "imagenet_robustbench-salman20-wrn-50-2",
            "imagenet_robustbench-salman20-rn-50",
        ],
        "corruption": [
            "imagenet_robustbench-tian22-deit-b",
            "imagenet_robustbench-tian22-deit-s",
            "imagenet_robustbench-erichson22-rn-50",
            "imagenet_robustbench-hendrycks20many-rn-50",
            "imagenet_robustbench-hendrycks20augmix-rn-50",
            "imagenet_robustbench-geirhos18-rn-50",
        ],
    },
}


class ExpManager:
    """Experiment manager for getting metadata."""

    def __init__(self) -> None:
        """Initialize ExpManager."""
        self._model_ids = {}
        for dataset, models in PUBLIC_MODELS.items():
            models_list = []
            for names in models.values():
                models_list.extend(names)
            self._model_ids[dataset] = models_list

        self._model_groups = {}
        for dataset, models in PUBLIC_MODELS.items():
            self._model_groups[dataset] = {}
            for group, names in models.items():
                for name in names:
                    self._model_groups[dataset][name] = group

    def _parse_model_name(self, dataset: str, model_name: str) -> str:
        if not model_name.startswith(dataset):
            model_name = f"{dataset}_{model_name}"
        if model_name not in self._model_ids[dataset]:
            raise KeyError(f"Given model_name ({model_name}) not found!")
        return model_name

    def get_model_id(self, dataset: str, model_name: str) -> int:
        """Get model id from model name."""
        model_name = self._parse_model_name(dataset, model_name)
        return self._model_ids[dataset].index(model_name)

    def get_model_group(self, dataset: str, model_name: str) -> str:
        """Get model group from model name.""" ""
        model_name = self._parse_model_name(dataset, model_name)
        return self._model_groups[dataset][model_name]

    def get_all_models(self, dataset: str) -> list[str]:
        """Get all model names."""
        return self._model_ids[dataset]

    def get_model_names(
        self, dataset: str, model_group: str | None = None
    ) -> list[str] | tuple[str, list[str]]:
        """Get model names from model group."""
        if model_group is None:
            return PUBLIC_MODELS[dataset]
        return PUBLIC_MODELS[dataset][model_group]


EXP_MANAGER = ExpManager()


HASH_TO_MODEL = {
    "7bf6": [
        "results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
    ],
    "b105": [
        "results/cifar10_huyvnphan-inception-v3/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
    ],
    "a696": [
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "b125": [
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "d6df": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "0e02": [
        "results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
    ],
    "8973": [
        "results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
    ],
    "4002": [
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step"
    ],
    "4262": [
        "results/cifar10_chenyaofo-repvgg-a0/saved_pgd_train_temp1.0_10step",
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "7834": [
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "f82e": [
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "b8f5": [
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "c7c7": [
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "ae03": [
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "50cc": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "7f06": [
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "656b": [
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "e295": [
        "results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "2947": [
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step"
    ],
    "a7f5": [
        "results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step",
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huyvnphan-inception-v3/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
    ],
    "21e6": [
        "results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
    ],
    "b029": [
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
    ],
    "c307": [
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "ee13": [
        "results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step",
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "bd0b": [
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "024a": [
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
    ],
    "230f": [
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
    ],
    "b320": [
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
    ],
    "7c2b": [
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step"
    ],
    "0dd1": [
        "results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
    ],
    "881f": [
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "53a1": [
        "results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "ed72": [
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "fcb6": [
        "results/cifar10_huyvnphan-densenet121/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "fada": [
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "a7e7": [
        "results/cifar10_chenyaofo-vgg11-bn/saved_pgd_train_temp1.0_10step",
        "results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "ea4a": [
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huyvnphan-inception-v3/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
    ],
    "6b67": [
        "results/cifar10_huyvnphan-densenet121/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
    ],
    "bb78": [
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
    ],
    "b076": [
        "results/cifar10_huyvnphan-densenet121/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
    ],
    "2102": [
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step"
    ],
    "d4fe": [
        "results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step",
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "4157": [
        "results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "cb89": [
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "b3d7": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "57ed": [
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "81aa": [
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "3650": [
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
    ],
    "729b": [
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step"
    ],
    "37c3": [
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step"
    ],
    "8e7b": [
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "32a9": [
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
    ],
    "dffd": [
        "results/cifar10_chenyaofo-repvgg-a0/saved_pgd_train_temp1.0_10step",
        "results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step",
        "results/cifar10_chenyaofo-vgg11-bn/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huyvnphan-inception-v3/saved_pgd_train_temp1.0_10step",
    ],
    "6b38": [
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "2d57": ["results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step"],
    "ee4a": [
        "results/cifar10_huyvnphan-inception-v3/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
    ],
    "c795": [
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step"
    ],
    "42c5": [
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step"
    ],
    "86ab": [
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "f0fa": [
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step"
    ],
    "3aa2": [
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step"
    ],
    "bc38": [
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "4a5a": [
        "results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "47a3": [
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huyvnphan-densenet121/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "8943": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
    ],
    "9e8c": [
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "04a9": [
        "results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "b48f": [
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
    ],
    "a9f3": [
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "d68a": [
        "results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_chenyaofo-vgg11-bn/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "18c6": [
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "7720": [
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "fe3f": [
        "results/cifar10_huyvnphan-inception-v3/saved_pgd_train_temp1.0_10step"
    ],
    "0382": [
        "results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step"
    ],
    "ad89": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huyvnphan-densenet121/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "33c5": [
        "results/cifar10_chenyaofo-repvgg-a0/saved_pgd_train_temp1.0_10step",
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
    ],
    "24f9": [
        "results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "6fce": [
        "results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "3f55": [
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step"
    ],
    "c2be": [
        "results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step",
        "results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
    ],
    "0631": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "b0d9": [
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step"
    ],
    "776a": [
        "results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step"
    ],
    "c5db": [
        "results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
    "6171": [
        "results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step",
        "results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "5657": [
        "results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step",
    ],
    "61d8": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
    ],
    "271c": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step",
    ],
    "d5b0": [
        "results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step",
        "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step",
    ],
}

HASH_TO_MODEL_ONE_EACH = {
    # "2207": ["results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step", "results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step", "results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step", "results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step"],
    # "1cc2": ["results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step", "results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step", "results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step", "results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step"],
    "e70b": ['results/cifar10_huyvnphan-inception-v3/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "507e": ['results/cifar10_chenyaofo-resnet20/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step'],
    "182b": ['results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step'],
    "e13c": ['results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step'],
    "f1ba": ['results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "ba42": ['results/cifar10_huyvnphan-densenet121/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "1f42": ['results/cifar10_chenyaofo-vgg11-bn/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "851a": ['results/cifar10_chenyaofo-vgg11-bn/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step'],
    "2283": ['results/cifar10_huyvnphan-densenet121/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step'],
    "2ee1": ['results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step'],
    "5d88": ['results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "e822": ['results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step'],
    "c563": ['results/cifar10_chenyaofo-repvgg-a0/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step'],
    "e62e": ['results/cifar10_huggingface-beit/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "6570": ['results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rony19-wrn-28-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step'],
    "f561": ['results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "a55f": ['results/cifar10_huggingface-vit-base/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "d6f2": ['results/cifar10_clip-RN50-laion/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step'],
    "ae98": ['results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "e713": ['results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step'],
    "0854": ['results/cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step'],
    "bb88": ['results/cifar10_chenyaofo-shufflenetv2-x0-5/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-sehwag20-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step'],
    "d35a": ['results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-augustin20-wrn-34-10-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-kireev21-rn-18/saved_pgd_train_temp1.0_10step'],
    "47b8": ['results/cifar10_chenyaofo-vgg11-bn/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd_train_temp1.0_10step'],
    "2bf8": ['results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-debenedetti22-xcit-l12/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step'],
    "99d9": ['results/cifar10_huggingface-convnext/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-xu23-wrn-28-10/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-modas21-rn-18/saved_pgd_train_temp1.0_10step'],
    "f337": ['results/cifar10_chenyaofo-mobilenetv2-x0-5/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-wang23-wrn-70-16/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd_train_temp1.0_10step', 'results/cifar10_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd_train_temp1.0_10step'],
}

HASH_TO_MODEL_ONE_EACH_RND_CIFAR100 = {
    "5478": ['cifar100_chenyaofo-vgg11-bn/saved_pgd', 'cifar100_robustbench-cui23-wrn-28-10/saved_pgd', 'cifar100_robustbench-modas21-rn-18/saved_pgd'],
    "44e1": ['cifar100_chenyaofo-mobilenetv2-x0-5/saved_pgd', 'cifar100_robustbench-jia22-wrn-34-20/saved_pgd', 'cifar100_robustbench-gowal20-wrn-70-16/saved_pgd'],
    "5c1e": ['cifar100_chenyaofo-mobilenetv2-x0-5/saved_pgd', 'cifar100_robustbench-jia22-wrn-34-20/saved_pgd', 'cifar100_robustbench-diffenderfer21-wrn-18-2/saved_pgd'],
    "1701": ['cifar100_inception-v3_local/saved_pgd', 'cifar100_robustbench-cui23-wrn-28-10/saved_pgd', 'cifar100_robustbench-gowal20-wrn-70-16/saved_pgd'],
    "0752": ['cifar100_chenyaofo-resnet20/saved_pgd', 'cifar100_robustbench-jia22-wrn-34-20/saved_pgd', 'cifar100_robustbench-addepalli22-wrn-34-10/saved_pgd'],
    "cb91": ['cifar100_densenet121_local/saved_pgd', 'cifar100_robustbench-jia22-wrn-34-20/saved_pgd', 'cifar100_robustbench-modas21-rn-18/saved_pgd'],
    "5648": ['cifar100_convmixer_local/saved_pgd', 'cifar100_robustbench-rade21-rn-18/saved_pgd', 'cifar100_robustbench-diffenderfer21-bin-wrn-18-2/saved_pgd'],
    "3a10": ['cifar100_chenyaofo-repvgg-a0/saved_pgd', 'cifar100_robustbench-debenedetti22-xcit/saved_pgd', 'cifar100_robustbench-addepalli22-wrn-34-10/saved_pgd'],
    "387f": ['cifar100_chenyaofo-resnet20/saved_pgd', 'cifar100_robustbench-bai23-mix/saved_pgd', 'cifar100_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd'],
    "0874": ['cifar100_chenyaofo-shufflenetv2-x0-5/saved_pgd', 'cifar100_robustbench-jia22-wrn-34-20/saved_pgd', 'cifar100_robustbench-gowal20-wrn-70-16/saved_pgd'],
    "795e": ['cifar100_chenyaofo-resnet20/saved_pgd', 'cifar100_robustbench-debenedetti22-xcit/saved_pgd', 'cifar100_robustbench-diffenderfer21-wrn-18-2/saved_pgd'],
    "5b39": ['cifar100_clip-RN50-laion/saved_pgd', 'cifar100_robustbench-jia22-wrn-34-20/saved_pgd', 'cifar100_robustbench-addepalli22-wrn-34-10/saved_pgd'],
    "92fd": ['cifar100_densenet121_local/saved_pgd', 'cifar100_robustbench-cui23-wrn-28-10/saved_pgd', 'cifar100_robustbench-modas21-rn-18/saved_pgd'],
    "5fde": ['cifar100_chenyaofo-mobilenetv2-x0-5/saved_pgd', 'cifar100_robustbench-rade21-rn-18/saved_pgd', 'cifar100_robustbench-diffenderfer21-wrn-18-2/saved_pgd'],
    "8635": ['cifar100_chenyaofo-repvgg-a0/saved_pgd', 'cifar100_robustbench-rade21-rn-18/saved_pgd', 'cifar100_robustbench-diffenderfer21-wrn-18-2/saved_pgd'],
    "fd6d": ['cifar100_chenyaofo-resnet20/saved_pgd', 'cifar100_robustbench-bai23-mix/saved_pgd', 'cifar100_robustbench-addepalli22-wrn-34-10/saved_pgd'],
    "e34f": ['cifar100_chenyaofo-resnet20/saved_pgd', 'cifar100_robustbench-wang23-wrn-70-16/saved_pgd', 'cifar100_robustbench-modas21-rn-18/saved_pgd'],
    "845d": ['cifar100_chenyaofo-repvgg-a0/saved_pgd', 'cifar100_robustbench-cui23-wrn-28-10/saved_pgd', 'cifar100_robustbench-modas21-rn-18/saved_pgd'],
    "69af": ['cifar100_huggingface-vit-base/saved_pgd', 'cifar100_robustbench-bai23-mix/saved_pgd', 'cifar100_robustbench-diffenderfer21-bin-wrn-18-2/saved_pgd'],
    "65fe": ['cifar100_senet18_local/saved_pgd', 'cifar100_robustbench-bai23-mix/saved_pgd', 'cifar100_robustbench-diffenderfer21-wrn-18-2/saved_pgd'],
    "840b": ['cifar100_densenet121_local/saved_pgd', 'cifar100_robustbench-debenedetti22-xcit/saved_pgd', 'cifar100_robustbench-gowal20-wrn-70-16/saved_pgd'],
    "d61b": ['cifar100_convmixer_local/saved_pgd', 'cifar100_robustbench-bai23-mix/saved_pgd', 'cifar100_robustbench-modas21-rn-18/saved_pgd'],
    "3cb2": ['cifar100_chenyaofo-vgg11-bn/saved_pgd', 'cifar100_robustbench-jia22-wrn-34-20/saved_pgd', 'cifar100_robustbench-modas21-rn-18/saved_pgd'],
    "d9db": ['cifar100_densenet121_local/saved_pgd', 'cifar100_robustbench-debenedetti22-xcit/saved_pgd', 'cifar100_robustbench-addepalli22-wrn-34-10/saved_pgd'],
    "6693": ['cifar100_convmixer_local/saved_pgd', 'cifar100_robustbench-debenedetti22-xcit/saved_pgd', 'cifar100_robustbench-modas21-rn-18/saved_pgd'],
    "e8a1": ['cifar100_densenet121_local/saved_pgd', 'cifar100_robustbench-bai23-mix/saved_pgd', 'cifar100_robustbench-addepalli22-wrn-34-10/saved_pgd'],
    "0c03": ['cifar100_chenyaofo-mobilenetv2-x0-5/saved_pgd', 'cifar100_robustbench-cui23-wrn-28-10/saved_pgd', 'cifar100_robustbench-addepalli22-wrn-34-10/saved_pgd'],
    "e7bd": ['cifar100_huggingface-swin-tiny/saved_pgd', 'cifar100_robustbench-wang23-wrn-70-16/saved_pgd', 'cifar100_robustbench-gowal20-wrn-70-16/saved_pgd'],
    "873f": ['cifar100_clip-RN50-laion/saved_pgd', 'cifar100_robustbench-rade21-rn-18/saved_pgd', 'cifar100_robustbench-hendrycks20-resnext-29-32x4d/saved_pgd'],
    "6cfd": ['cifar100_senet18_local/saved_pgd', 'cifar100_robustbench-rade21-rn-18/saved_pgd', 'cifar100_robustbench-diffenderfer21-wrn-18-2/saved_pgd'],
}

HASH_TO_MODEL_3GROUPS_CIFAR10 = {
    "ba3a": ['cifar10_huggingface-beit/saved_pgd', 'cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd'],
    "8c53": ['cifar10_huggingface-beit/saved_pgd', 'cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd'],
    "4cdb": ['cifar10_huggingface-beit/saved_pgd', 'cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd'],
    "e0d6": ['cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd', 'cifar10_robustbench-xu23-wrn-28-10/saved_pgd'],
    "41c0": ['cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd', 'cifar10_robustbench-xu23-wrn-28-10/saved_pgd'],
    "bdaa": ['cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd', 'cifar10_robustbench-xu23-wrn-28-10/saved_pgd'],
    "49b1": ['cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd'],
    "78af": ['cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd'],
    "431a": ['cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-rebuffi21-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd'],
    "6f63": ['cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-kireev21-rn-18/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd'],
    "65b7": ['cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-kireev21-rn-18/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16-l2/saved_pgd'],
    "4fcc": ['cifar10_huggingface-vit-base/saved_pgd', 'cifar10_robustbench-diffenderfer21-wrn-18-2/saved_pgd', 'cifar10_robustbench-kireev21-rn-18/saved_pgd', 'cifar10_robustbench-wang23-wrn-70-16/saved_pgd'],
}