from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split


def data_transforms():
      train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Imagenet mean and std for normalization
      ])

      # Transformation for validation data
      val_transforms = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      return train_transforms, val_transforms
  

def dataloaders(train_data_dir, split=0.2, seed=42, batch_size=32):
    
    train_transforms, val_transforms = data_transforms()
    
    train_all = datasets.ImageFolder(train_data_dir, transform=train_transforms)
    targets = train_all.targets 
    indices = list(range(len(train_all)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=split,
        stratify=targets,   
        random_state=seed
    )
    # split the dataset into training and validation subsets
    train_dataset = Subset(train_all, train_idx)
    val_trans = datasets.ImageFolder(train_data_dir, transform=val_transforms)
    val_dataset   = Subset(val_trans, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader