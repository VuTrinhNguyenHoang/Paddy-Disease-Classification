from torch.utils.data import Dataset
from PIL import Image

class PaddyDataset(Dataset):
    def __init__(self, df, transforms, label2id):
        self.df = df
        self.tfm = transforms
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["path"]
        label = self.label2id[row["label"]] if self.label2id != None else -1
            
        img = Image.open(image_path).convert("RGB")

        return self.tfm(img), label