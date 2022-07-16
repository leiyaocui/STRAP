import os
import yaml
from PIL import Image
import pickle
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_dir, phase, transforms):
        self.transforms = transforms

        self.image_list = []
        self.label_list = []
        self.file_name_list = []
        self.keypoint_list = []
        self.with_keypoint = False
        
        keypoint_dict = None
        keypoint_path = os.path.join(data_dir, phase + "_keypoint.yaml")
        if os.path.exists(keypoint_path):
            with open(keypoint_path, "r") as fb:
                keypoint_dict = yaml.safe_load(fb)
        
        if keypoint_dict is not None:
            self.with_keypoint = True

        for line in open(os.path.join(data_dir, phase + ".txt"), "r"):
            image_path, label_path = line.strip().split(",")
            
            file_name = os.path.basename(image_path).split(".")[0]
            self.file_name_list.append(file_name)
            
            self.image_list.append(os.path.join(data_dir, image_path))
            self.label_list.append(os.path.join(data_dir, label_path))
            
            if self.with_keypoint:
                self.keypoint_list.append(keypoint_dict[file_name])

        assert len(self.image_list) == len(self.label_list)

    def __getitem__(self, index):
        data = dict()

        data["file_name"] = self.file_name_list[index]

        data["image"] = Image.open(self.image_list[index])
        
        with open(self.label_list[index], "rb") as fb:
            dense_label = pickle.load(fb)
            data["dense_label"] = [Image.fromarray(dense_label[:, :, i]) for i in range(dense_label.shape[2])]
        
        if self.with_keypoint:
            data["point_label"] = self.keypoint_list[index]

        data["validity"] = Image.new("L", data["image"].size, color=1)

        data = self.transforms(data)

        if self.with_keypoint:
            del data["point_label"]
        
        return data

    def __len__(self):
        return len(self.image_list)

def make_dataloader(data_dir, phase, transforms, **aargs):
    dataset = CustomDataset(data_dir, phase, transforms)
    
    return DataLoader(dataset, **aargs)