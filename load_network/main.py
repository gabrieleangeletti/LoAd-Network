import argparse
import logging

import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import tqdm

import dataset
import load

logging.getLogger().setLevel(logging.INFO)


def main(
    dataset_dir: str,
    base_net: str,
    model_path: str,
) -> None:
    batch_size = 8

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: np.array(x)),
        transforms.Lambda(lambda x: x[:, :, [2, 1, 0]]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.mul(x, 255)),
        transforms.Normalize(
            mean=[103.939, 116.779, 123.68],
            std=[1.0, 1.0, 1.0],
        ),
    ])

    logging.info("Loading dataset: %s", dataset_dir)
    img_dataset = dataset.LoadDataset(root=dataset_dir, base_net=base_net, transform=preprocess)
    loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logging.info("loading model: %s", base_net)
    model = load.LoadNetwork("vgg16", (3,), (3,), 15)
    logging.info("Loading state dict: %s", model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    pbar = tqdm.tqdm(loader, desc="Evaluating model", total=len(img_dataset) // batch_size)
    running_corrects = 0
    for i, (input_imgs, input_maps, labels) in enumerate(pbar):
        if torch.cuda.is_available():
            input_imgs = Variable(input_imgs.cuda())
            input_maps = Variable(input_maps.cuda())
            labels = Variable(labels.cuda())
        else:
            input_imgs = Variable(input_imgs)
            input_maps = Variable(input_maps)
            labels = Variable(labels)

        outputs = model((input_imgs, input_maps))
        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data)

        if i > 0 and i % 10 == 0:
            running_acc = running_corrects / ((i + 1) * batch_size)
            pbar.set_description("Accuracy after {:d} batches: {:.2f}".format(i + 1, running_acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
    parser.add_argument("--base-net", type=str, help="Base network ('alexnet' or 'vgg16')")
    parser.add_argument("--model-path", type=str, help="Model state dict .pth path.")
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        base_net=args.base_net,
        model_path=args.model_path,
    )
