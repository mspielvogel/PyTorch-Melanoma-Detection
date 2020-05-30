import os
import time
import torch
from torchvision import transforms, models

from Datasets import MoleDataset


class CNNTrainer():
    def __init__(self, config):
        self._input_folder = config.get("input_folder", "data/training_sets/HAM10000")
        self._batchsize = config.get("batch_size", 16)
        self._epochs = config.get("epochs", 50)
        self._val_every_epoch = config.get("val_every_epoch", 5)
        learning_rate = config.get("learning_rate", 0.001)
        learning_rate_decay = config.get("learning_rate_decay", 0.997)

        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor()])

        train_dataset = MoleDataset(input_folder=os.path.join(self._input_folder, "train"),
                                    csv_file=os.path.join(self._input_folder, "labels.csv"),
                                    transform=transform)

        val_dataset = MoleDataset(input_folder=os.path.join(self._input_folder, "validation"),
                                  csv_file=os.path.join(self._input_folder, "labels.csv"),
                                  transforms=transforms)

        self._train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self._batchsize,
                                                         shuffle=True, num_workers=4)
        self._val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self._batchsize,
                                                       shuffle=False, num_workers=4)

        self._model = models.resnet18(pretrained=True)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, learning_rate_decay)

        if torch.cuda.is_available():
            self._use_gpu = True
        else:
            self._use_gpu = False

        if self._use_gpu:
            self._model.cuda()

    def train(self):
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
                self._scheduler.step()

                running_loss += loss.item()
            train_loss = running_loss / i
            print("Epoch: {}/{}\nLearning Rate: {}\nTraining loss: {:03.4f}".format(epoch, self._epochs,
                                                                                    self._scheduler.get_last_lr(),
                                                                                    train_loss))

            if epoch % self._val_every_epoch == 0:
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

            print("Time: {:03.2f}\n\n".format(time.time() - start))

        example = torch.rand(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(self._model, example)
        traced_script_module.save("app/src/main/assets/model.pt")


if __name__ == "__main__":
    config = {"input_folder": "data/training_sets/HAM10000",
              "batch_size": 16,
              "epochs": 10,
              "val_every_epoch": 2,
              "learning_rate": 0.001,
              "learning_rate_decay": 0.997}

    trainer = CNNTrainer(config)
    trainer.train()
