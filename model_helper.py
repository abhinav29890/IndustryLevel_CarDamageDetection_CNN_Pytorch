from pathlib import Path

import torch
from torch import nn
from torchvision import models
from torchvision.io import read_image

trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']
MODEL_PATH = Path(__file__).resolve().parent / "model" / "saved_model.pth"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights=None)
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def _prepare_image(image_path):
    image = read_image(str(image_path)).float()
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] != 3:
        image = image[:3]

    image = image / 255.0
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    image = (image - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return image.unsqueeze(0)


def predict(image_path):
    image_tensor = _prepare_image(image_path)

    global trained_model

    if trained_model is None:
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        trained_model.to("cpu")
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
