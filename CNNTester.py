import os
import sklearn.metrics as metrics
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from Datasets import MoleDataset


class CNNTester():
    def __init__(self, config):
        self._input_folder = config["input_folder"]
        model_path = config["model"]
        self._model = nn.Sequential(models.mobilenet_v2(), nn.Linear(1000, 7))
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()
        #self._output_path = config["output_path"]

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(224),
                                        transforms.ToTensor()])
        test_set = MoleDataset(input_folder=os.path.join(self._input_folder, "test"),
                                    csv_file=os.path.join(self._input_folder, "labels.csv"),
                                    transform=transform)
        self._names = test_set.names

        self._test_loader = torch.utils.data.DataLoader(test_set, batch_size=16,
                                                        shuffle=False, num_workers=4)

        if torch.cuda.is_available():
            self._use_gpu = True
        else:
            self._use_gpu = False

        if self._use_gpu:
            self._model.cuda()


    def test(self):

        y_pred = np.array([])
        y_true = np.array([])
        for i, batch in enumerate(self._test_loader):
            inputs, labels = batch
            if self._use_gpu:
                inputs = inputs.cuda()

            preds = np.argmax(self._model(inputs).detach().cpu().numpy(), axis=1)
            y_pred = np.concatenate((y_pred, preds))
            y_true = np.concatenate((y_true, labels.numpy()))

        print(metrics.classification_report(y_true, y_pred, target_names=self._names))


if __name__ == "__main__":
    config = {"input_folder": "data/training_sets/HAM10000",
              "model": "models/mobilenet_v2_2020_05_31.pth",
              "batch_size": 16,
              "epochs": 10,
              "val_every_epoch": 2,
              "learning_rate": 0.001,
              "learning_rate_decay": 0.997}

    tester = CNNTester(config)
    tester.test()