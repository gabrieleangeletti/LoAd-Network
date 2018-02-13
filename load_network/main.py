import argparse

import torch
import torchvision.transforms as transforms

from . import dataset
from . import load


def main(
    dataset_dir: str,
    base_net: str,
) -> None:
    preprocess = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img_dataset = dataset.LoadDataset(root=dataset_dir, base_net=base_net, transform=preprocess)
    torch.utils.data.DataLoader(img_dataset, batch_size=4, shuffle=True, num_workers=4)
    load.LoadNetwork("vgg16", (3,), (3,), 15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset-dir", type=str, help="Dataset directory")
    parser.add_argument("base-net", type=str, help="Base network ('alexnet' or 'vgg16')")
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        base_net=args.base_net,
    )
