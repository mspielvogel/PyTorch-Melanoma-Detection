import torch
import torch.nn as nn
from torchvision import models


# model = models.resnet18(pretrained=True)
model = nn.Sequential(models.mobilenet_v2(pretrained=False), nn.Linear(1000, 7))
model.load_state_dict(torch.load("models/mobilenet_v2_2020_05_31.pth", map_location={'cuda:0': 'cpu'}))
model.eval()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("MoleDetectionApp/app/src/main/assets/resnet18_s.pt")
traced_script_module.save("MoleDetectionApp/app/src/main/assets/mobilenet_v2.pt")
