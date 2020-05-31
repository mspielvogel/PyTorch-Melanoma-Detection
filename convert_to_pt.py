import torch
from torchvision import models


model = models.resnet18(pretrained=True)
# model.load_state_dict(torch.load("models/mobilenet_v2_2020_05_30.pth"))
model.eval()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("MoleDetectionApp/app/src/main/assets/resnet18_s.pt")