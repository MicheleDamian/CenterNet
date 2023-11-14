import torch

from torch.nn import Module, functional as F
from lib.models.networks.msra_resnet import BasicBlock, PoseResNet


class Inference(Module):

    def __init__(self, mean, std, model_path, downsample=4, num_predictions=10, device='cuda'):
        super().__init__()

        self.mean = mean
        self.std = std
        self.downsample = downsample
        self.num_predictions = num_predictions
        self.device = device

        self.model = PoseResNet(
            BasicBlock,
            [2, 2, 2, 2],
            {'hm': 1, 'wh': 2, 'reg': 2},
            head_conv=64
        )
        self.model.init_weights(18, pretrained=True)

        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self, x: torch.Tensor):

        def pad(d): return (d + 31) // 32 * 32

        height, width = x.shape[:2]

        height_pad, width_pad = pad(height), pad(width)
        width_ds = width_pad // self.downsample

        x -= self.mean
        x /= self.std

        x = x.permute(2, 0, 1)

        pad_h, pad_w = height_pad - height, width_pad - width
        padding = (
            pad_w // 2, pad_w // 2 + pad_w % 2,
            pad_h // 2, pad_h // 2 + pad_h % 2
        )
        x = F.pad(x, padding).unsqueeze(0)

        output = self.model(x)[0]

        center = F.sigmoid(output['hm'])
        width_height = output['wh'].squeeze()
        offset = output['reg'].squeeze()

        center_max = F.max_pool2d(center, (3, 3), stride=1, padding=1)
        center = torch.where(center_max == center, center, torch.zeros(1).to(self.device)).squeeze()

        score, idx = torch.topk(center.view(-1), self.num_predictions)

        offset = offset.view(2, -1)[:, idx]
        width_height = width_height.view(2, -1)[:, idx] / 2

        xs, ys = (idx % width_ds) + offset[0], (idx // width_ds) + offset[1]

        detections = torch.stack(
            (xs - width_height[0], ys - width_height[1], xs + width_height[0], ys + width_height[1], score),
            dim=1
        )

        detections[:, :4] *= self.downsample
        detections[:, 0] -= pad_w // 2
        detections[:, 1] -= pad_h // 2
        detections[:, 2] -= pad_w // 2
        detections[:, 3] -= pad_h // 2

        return detections
