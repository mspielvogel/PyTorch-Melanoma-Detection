import sklearn.metrics
import torch
import os
from torchvision import transforms, models

from Datasets import MoleDataset


class CNNTester():
    def __init__(self, config):
        self._input_folder = config["input_folder"]
        model_path = config["model"]
        self._model = models.mobilenet_v2()
        self._model = self._model.load_state_dict(torch.load(model_path))
        self._model.eval()
        self._output_path = config["output_path"]

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(224),
                                        transforms.ToTensor()])
        test_set = MoleDataset(input_folder=os.path.join(self._input_folder, "test"),
                                    csv_file=os.path.join(self._input_folder, "labels.csv"),
                                    transform=transform)


    def test(self):

        y_pred = []
        y_true = []
        for i, batch in enumerate(self._test_loader):
            inputs, labels = batch
            if self._use_gpu:
                inputs = inputs.cuda()

            outputs = self._model(inputs).detach().cpu().numpy().tolist()
            y_pred.extend(outputs)
            y_true.extend(labels)





if __name__ == "__main__":
    config = {"input_folder": "data/training_sets/HAM10000",
              "output_model": "models/mobilenet_v2_2020_05_30.pth",
              "batch_size": 16,
              "epochs": 10,
              "val_every_epoch": 2,
              "learning_rate": 0.001,
              "learning_rate_decay": 0.997}

    tester = CNNTester(config)
    tester.test()