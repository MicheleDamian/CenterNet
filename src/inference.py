import torch

from torch.nn import functional as F
from .lib.models.networks.msra_resnet import BasicBlock, PoseResNet


class Inference:

    def __init__(self, mean, std, model_path, device='gpu'):

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

        in_height, in_width = height | 32, width | 32

        image = image.to(self.device)

        image -= self.mean
        image /= self.std

        pad_h, pad_w = in_height - height, in_width - width
        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
        image = F.padding(image, padding)

        image = image.transpose(2, 0, 1).unsqueeze(0)

        print('image.shape', image.shape)
        print('padding', padding)

        output = self.model(image)[-1]

        print('output.shape', output.shape)

        hm = F.sigmoid(output['hm'].squeeze())
        wh = output['wh'].squeeze()
        reg = output['reg'].squeeze()

        print('hm.shape', hm.shape)
        print('wh.shape', wh.shape)
        print('reg.shape', reg.shape)

        hm_max = F.max_pool2d(hm, (3, 3), padding=1)
        hm = torch.where(hm_max == hm, hm, torch.zeros(1))

        print('hm_max.shape', hm_max.shape)
        print('hm.shape', hm.shape)

        score, idx = torch.topk(hm.view(-1), k)

        print('score.shape', score.shape)

        reg = reg.view(2, -1)[:, idx]
        wh = wh.view(2, -1)[:, idx] / 2

        print('reg.shape', reg.shape)
        print('wh.shape', wh.shape)

        xs = (idx % width).floor() + reg[0] - pad_w // 2
        ys = (idx / width).floor() + reg[1] - pad_h // 2

        print('xs.shape', xs.shape)

        detections = torch.cat([xs - wh[0], ys - wh[1], xs + wh[0], ys + wh[1], score])

        print('detections.shape', detections.shape)

        return detections.detach().cpu().numpy()
