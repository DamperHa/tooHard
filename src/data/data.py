import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data_list = self.load_data_list()

    def load_data_list(self):
        list_file = os.path.join(self.root, f'{self.split}.txt')
        with open(list_file, 'r') as f:
            data_list = [line.strip() for line in f.readlines()]
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        pc_file = os.path.join(self.root, self.split, data_name, 'point_cloud.bin')
        img_file = os.path.join(self.root, self.split, data_name, 'image.jpg')
        label_pc_file = os.path.join(self.root, self.split, data_name, 'label_pc.txt')
        label_img_file = os.path.join(self.root, self.split, data_name, 'label_img.txt')

        # Load point cloud data
        pc_data = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)

        # Load image data
        img_data = Image.open(img_file).convert('RGB')

        # Load point cloud and image labels
        label_pc = np.loadtxt(label_pc_file, dtype=np.int64)
        label_img = np.loadtxt(label_img_file, dtype=np.int64)

        # Apply transformations if specified
        if self.transform is not None:
            pc_data = self.transform['pc'](pc_data)
            img_data = self.transform['img'](img_data)

        return {'pc': pc_data, 'img': img_data, 'label_pc': label_pc, 'label_img': label_img}


# Define transformations (you can customize these)
transform = {
    'pc': transforms.Compose([
        # Add your point cloud transformations here if needed
    ]),
    'img': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),           # Convert to tensor
        # Add other image transformations here if needed
    ])
}

# Create datasets and dataloaders
# train_dataset = CustomDataset(root='your_root_path', split='train', transform=transform)
# val_dataset = CustomDataset(root='your_root_path', split='val', transform=transform)
#
# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
# val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
