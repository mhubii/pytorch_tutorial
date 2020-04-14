import cv2
from PIL import Image
import pandas as pd
from torch.utils.data.dataset import Dataset

class ImageLabel(Dataset):
    def __init__(self, df_loc, transforms=None):
        self.df = pd.read_pickle(df_loc)
        self.transforms = transforms

    def __getitem__(self, idx):
        img = cv2.imread(self.df['file_path'][idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transforms:
            img = self.transforms(img)

        label = self.df['label'][idx]
        return img, label

    def __len__(self):
        return len(self.df)
