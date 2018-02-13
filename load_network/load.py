from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

BASE_NETWORKS = dict(
    alexnet=dict(
        init=models.alexnet,
        depth=256,
        fc6=2048,
        fc7=2048,
    ),
    vgg16=dict(
        init=models.vgg16,
        depth=512,
        fc6=4096,
        fc7=2048,
    )
)


class LoadNetwork(nn.Module):

    def __init__(
        self,
        base_net: str,
        spp_conv5: Tuple[int],
        spp_mask: Tuple[int],
        num_classes: int,
    ) -> None:
        assert isinstance(base_net, str) and base_net in BASE_NETWORKS.keys()
        assert isinstance(spp_conv5, tuple) and all([isinstance(s, int) for s in spp_conv5])
        assert isinstance(spp_mask, tuple) and all([isinstance(s, int) for s in spp_mask])
        assert isinstance(num_classes, int)
        super(LoadNetwork, self).__init__()

        base_net = BASE_NETWORKS[base_net]
        spp_dim = sum([base_net["depth"] * s * s for s in spp_conv5 + spp_mask])

        model = base_net["init"](pretrained=True)
        model.features = nn.Sequential(*list(model.features.children())[:-1])

        model.spp_conv5 = (nn.AdaptiveMaxPool2d(output_size=s) for s in spp_conv5)
        model.spp_mask = (nn.AdaptiveMaxPool2d(output_size=s) for s in spp_mask)

        model.classifier = nn.Sequential(
            nn.Linear(spp_dim, base_net["fc6"]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(base_net["fc6"], base_net["fc7"]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(base_net["fc7"], num_classes),
        )

        self.model = model

    def forward(
        self,
        input_: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        img, mask = input_

        img_features = self.model.features(img)
        msk_features = torch.mul(img_features, mask)

        img_pooled = torch.cat((spp(img_features) for spp in self.model.spp_conv5), 2)
        msk_pooled = torch.cat((spp(msk_features) for spp in self.model.spp_mask), 2)

        fc_input = torch.cat((img_pooled, msk_pooled), 2)

        return self.model.classifier(fc_input)
