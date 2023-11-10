import torch

from torch.nn import functional as F
from lib.models.networks.msra_resnet import BasicBlock, PoseResNet


class Inference:

    def __init__(self, mean, std, model_path, device='cuda'):
        self.mean = mean
        self.std = std
        self.device = device

        heads = {'hm': 1, 'wh': 2, 'reg': 2}

        head_conv = 64

        block_class, layers = BasicBlock, [2, 2, 2, 2]
        model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
        model.init_weights(18, pretrained=True)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

        self.model = model.to(device)
        self.model.eval()

    def run(self, image, k=10):

        height, width = image.shape[:2]

        in_height, in_width = (height | 31) + 1, (width | 31) + 1

        image -= self.mean[None]
        image /= self.std[None]

        image = image.permute(2, 0, 1)

        image = image.to(self.device)

        print('image.shape', image.shape)

        pad_h, pad_w = in_height - height, in_width - width
        padding = (
            pad_w // 2, pad_w // 2 + pad_w % 2,
            pad_h // 2, pad_h // 2 + pad_h % 2
        )
        image = F.pad(image, padding)

        print('image.shape', image.shape)

        image = image.unsqueeze(0).type(torch.float32)

        print('image.shape', image.shape)
        print('padding', padding)

        output = self.model(image)[0]

        hm = F.sigmoid(output['hm'])
        wh = output['wh'].squeeze()
        reg = output['reg'].squeeze()

        print('hm.shape', hm.shape)
        print('wh.shape', wh.shape)
        print('reg.shape', reg.shape)

        hm_max = F.max_pool2d(hm, (3, 3), stride=1, padding=1)

        print('hm_max.shape', hm_max.shape)

        hm = torch.where(hm_max == hm, hm, torch.zeros(1).to(self.device)).squeeze()

        print('hm.shape', hm.shape)

        score, idx = torch.topk(hm.view(-1), k)

        print('score.shape', score.shape)

        reg = reg.view(2, -1)[:, idx]
        wh = wh.view(2, -1)[:, idx] / 2

        print('reg.shape', reg.shape)
        print('wh.shape', wh.shape)

        xs = (idx % in_width).floor() + reg[0]
        ys = (idx / in_width).floor() + reg[1]

        print('xs.shape', xs.shape)

        detections = torch.stack(
            (xs - wh[0], ys - wh[1], xs + wh[0], ys + wh[1], score),
            dim=1
        )

        detections[:4] *= 4
        detections[0] -= pad_w // 2
        detections[1] -= pad_h // 2
        detections[2] -= pad_w // 2
        detections[3] -= pad_h // 2

        print('detections.shape', detections.shape)

        return detections.detach().cpu().numpy()
