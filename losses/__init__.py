import torch
import torch.nn as nn
import torchvision.models as models

def L2_norm(image_rec, image_orig):
    assert image_rec.shape == image_orig.shape, "Assertion error: shape of input should be as same as target"

    return torch.linalg.norm(image_rec.reshape(image_rec.shape[0], -1) - image_orig.reshape(image_orig.shape[0], -1), ord=2, dim=1).mean()


def L1_norm(encoder,encoder_rec):
    assert encoder.shape == encoder_rec.shape, "Assertion error: shape of input should be as same as target"

    return torch.linalg.norm(encoder.reshape(encoder.shape[0], -1) - encoder_rec.reshape(encoder_rec.shape[0], -1), ord=1, dim=1).mean()

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=None, weights=None, device="cpu"):
        super(VGGPerceptualLoss, self).__init__()

        if layers is None:
            layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        if weights is None:
            weights = [1.0] * len(layers)

        self.device = device
        self.layers = layers
        self.weights = weights

        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.vgg = vgg

        self.layer_name_mapping = {
            'relu1_1': 1,
            'relu1_2': 3,
            'relu2_1': 6,
            'relu2_2': 8,
            'relu3_1': 11,
            'relu3_2': 13,
            'relu3_3': 15,
            'relu3_4': 17,
            'relu4_1': 20,
            'relu4_2': 22,
            'relu4_3': 24,
            'relu4_4': 26,
            'relu5_1': 29,
            'relu5_2': 31,
            'relu5_3': 33,
            'relu5_4': 35
        }

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):

        x = self.normalize_batch(x)
        y = self.normalize_batch(y)

        loss = 0.0

        x_input = x
        y_input = y

        for i, layer in enumerate(self.layers):
            x_feat = self.extract_features(x_input.clone(), self.layer_name_mapping[layer])
            y_feat = self.extract_features(y_input.clone(), self.layer_name_mapping[layer])
            loss += self.weights[i] * nn.functional.l1_loss(x_feat, y_feat)

        return loss

    def extract_features(self, x, end_layer):
        for i in range(end_layer + 1):
            x = self.vgg[i](x)
        return x

    def normalize_batch(self, batch):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        assert batch.shape[1] == 3, f"Expected 3-channel input, got {batch.shape[1]}"
        return (batch - mean) / std
