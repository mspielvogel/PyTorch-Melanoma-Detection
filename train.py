import torch

from Datasets import MoleDataset


train_dataset = MoleDataset(input_folder='data/training_sets/HAM10000/train/',
                            csv_file='data/training_sets/HAM10000/HAM10000_metadata.csv')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

for i, batch in enumerate(train_loader):
    import pdb; pdb.set_trace()