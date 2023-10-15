import itertools
import pickle
from pathlib import Path

# Cosine similarity
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from exp_params import ATTACKS, EXP_MANAGER, PUBLIC_MODELS

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.CIFAR10(
    root="../data", train=False, transform=transform, download=True
)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

NUM_SAMPLES = 10000
dataset = "cifar10"
idx = np.arange(len(test_dataset))
np.random.shuffle(idx)
idx = idx[:NUM_SAMPLES]

all_models = []
for models in PUBLIC_MODELS[dataset].values():
    all_models.extend(models)

ignore_atk_idx = 226
adv_idx_dict = {}
for i, (src, atk) in enumerate(itertools.product(all_models, ATTACKS)):
    if i == ignore_atk_idx:
        continue
    if i > ignore_atk_idx:
        i -= 1
    adv_idx_dict[i] = (src, atk)
    if "diffender" in src and atk == "naa":
        print(i, src, atk)

cos = np.zeros((len(adv_idx_dict), len(adv_idx_dict), len(idx)))

for i, index in enumerate(idx):
    if i % 10 == 0:
        print(i)

    clean_image, y = test_dataset[index]
    base_path = Path("results")
    adv_images = []

    # Load all the adversarial examples
    # for model in all_models:
    #     for attack in ATTACKS:
    for atk_idx in range(len(adv_idx_dict)):
        model, attack = adv_idx_dict[atk_idx]
        tmp_path = (
            base_path
            / model
            / f"saved_{attack}_test_temp1.0"
            / f"{y:05d}"
            / f"{index:05d}_00.png"
        )
        image = np.array(Image.open(tmp_path)) / 255
        adv_images.append(image.transpose(2, 0, 1).flatten())
    adv_images = np.stack(adv_images, axis=0)

    adv_pert = adv_images - clean_image.numpy().flatten()

    cos[:, :, i] = cosine_similarity(adv_pert)


with open("cosine_similarity.pkl", "wb") as f:
    pickle.dump(cos, f)
