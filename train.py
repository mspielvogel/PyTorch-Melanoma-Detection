import torch
import torchvision

from Datasets import MoleDataset

transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = MoleDataset(input_folder='data/training_sets/HAM10000/train/',
                            csv_file='data/training_sets/HAM10000/HAM10000_metadata.csv',
                            transforms=transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

model = torchvision.models.resnet18(pretrained=True)
model.train()

for i, batch in enumerate(train_loader):
    import pdb; pdb.set_trace()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("app/src/main/assets/model.pt")