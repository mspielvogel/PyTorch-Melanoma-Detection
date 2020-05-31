import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models

from Datasets import MoleDataset


class CNNTrainer():
    def __init__(self, config):
        self._input_folder = config.get("input_folder", "data/training_sets/HAM10000")
        self._output_model = config["output_model"]
        self._batchsize = config.get("batch_size", 16)
        self._epochs = config.get("epochs", 50)
        self._val_every_epoch = config.get("val_every_epoch", 5)
        pretrain_model = config.get("pretrain_model", None)
        learning_rate = config.get("learning_rate", 0.001)
        learning_rate_decay = config.get("learning_rate_decay", 0.997)
        num_classes = config.get("num_classes", 7)

        if not os.path.exists(os.path.dirname(self._output_model)):
            os.makedirs(os.path.dirname(self._output_model))

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.3),
                                        transforms.RandomGrayscale(p=0.1),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor()])

        train_dataset = MoleDataset(input_folder=os.path.join(self._input_folder, "train"),
                                    csv_file=os.path.join(self._input_folder, "labels.csv"),
                                    transform=transform)

        val_dataset = MoleDataset(input_folder=os.path.join(self._input_folder, "validation"),
                                  csv_file=os.path.join(self._input_folder, "labels.csv"),
                                  transform=transform)

        self._train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self._batchsize,
                                                         shuffle=True, num_workers=4)
        self._val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self._batchsize,
                                                       shuffle=False, num_workers=4)

        self._model = nn.Sequential(models.mobilenet_v2(pretrained=True), nn.Linear(1000, num_classes))
        if pretrain_model is not None:
            self._model.load_state_dict(torch.load(pretrain_model))

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, learning_rate_decay)
        self._loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self._use_gpu = True
        else:
            self._use_gpu = False

        if self._use_gpu:
            self._model.cuda()

    def train(self):
        best_val_loss = 1000000
        for epoch in range(self._epochs):
            start = time.time()

            # Train trun
            running_loss = 0
            self._model.train()
            for i, batch in enumerate(self._train_loader):
                inputs, labels = batch
                if self._use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self._loss(outputs, labels)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.item()
            train_loss = running_loss / i
            print("Epoch: {}/{}\nLearning Rate: {}\nTraining loss: {:03.4f}".format(epoch + 1, self._epochs,
                                                                                    self._scheduler.get_lr()[0],
                                                                                    train_loss))
            self._scheduler.step()

            if (epoch + 1) % self._val_every_epoch == 0:
                # Validation run
                running_loss = 0
                self._model.eval()
                for i, batch in enumerate(self._val_loader):
                    inputs, labels = batch
                    if self._use_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    outputs = self._model(inputs)
                    loss = self._loss(outputs, labels)

                    running_loss += loss.item()
                val_loss = running_loss / i
                print("Validation loss: {:03.4f}".format(val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"\nSaving best model: {self._output_model}\n")
                    torch.save(self._model.state_dict(), self._output_model)

            print("Time: {:03.2f}\n\n".format(time.time() - start))




if __name__ == "__main__":
    config = {"input_folder": "data/training_sets/HAM10000",
              "output_model": "models/mobilenet_v2_2020_05_31.pth",
              "pretrain_model": "models/mobilenet_v2_2020_05_30.pth",
              "batch_size": 16,
              "epochs": 20,
              "val_every_epoch": 1,
              "learning_rate": 0.001,
              "learning_rate_decay": 0.997,
              "num_classes": 7}

    trainer = CNNTrainer(config)
    trainer.train()
