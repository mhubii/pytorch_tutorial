import cv2
import PIL
import pandas as pd
from torch.utils.data.dataset import Dataset

class ImageLabel(Dataset):
    def __init__(self, df_loc):
        self.df = pd.read_pickle(df_loc)

    def __getitem__(self, idx):
        img = cv2.imread(self.df['file_path'][idx])
        label = self.df['label'][idx]

        return img, label

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    image_dataset = ImageLabel('/tmp/pytorch_tutorial/files.df')
    img, _ = image_dataset.__getitem__(0)
    cv2.imshow('img',img)
    cv2.waitKey()
