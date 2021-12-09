import tarfile
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(transforms=transforms.ToTensor(), batch_size=32, shuffle=True, num_workers=2):
    """
    Extract the MiDaS-60 tar file and return dataset and data loader objects for the train and test sets, respectively.
    """
    tar = tarfile.open('MiDaS-60.tar.001')
    tar.extractall('./root')
    tar.close()
    
    data_folder = os.listdir('./root')[0]
    
    train_dataset = datasets.ImageFolder(os.path.join(f'root/{data_folder}', 'train'), transform=transforms)
    test_dataset = datasets.ImageFolder(os.path.join(f'root/{data_folder}', 'test'), transform=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader, train_dataset, test_loader, test_dataset