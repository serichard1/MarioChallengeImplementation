import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import pandas as pd

class TwinSet(Dataset):

    def __init__(self, root, transforms, split='train'):
        super().__init__()

        self.full_dataset = pd.read_csv(os.path.join(root, f'all_data/df_{split}.csv'))
        self.instances = self.full_dataset['case']
        _, counts = np.unique(self.full_dataset['label'], return_counts=True)
        self.class_weights = 1. / torch.tensor(counts, dtype=torch.float).cuda(non_blocking=True)
        self.root = './MARIO_challenge/train/'
        self.transforms = transforms
  
    def _import_img(self, img_path):
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    def _get_img_path(self, case_data, variable, tj=True):
        ti_path = os.path.join(self.root, case_data[variable].item())
        if not tj:
            return ti_path
        tj_path = os.path.join(self.root, case_data[f'{variable}+1'].item())
        return ti_path, tj_path

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        case_id = self.instances[idx]
        case_data = self.full_dataset[self.full_dataset['case'] == case_id]

        bscan_ti_path, bscan_tj_path = self._get_img_path(case_data, 'image_at_ti')
        localizer_ti_path = self._get_img_path(case_data, 'LOCALIZER', tj=False)

        bscan_ti, bscan_tj = self._import_img(bscan_ti_path), self._import_img(bscan_tj_path)
        localizer_ti = self._import_img(localizer_ti_path)

        bscan_ti, bscan_tj = self.transforms(bscan_ti), self.transforms(bscan_tj)
        localizer_ti = self.transforms(localizer_ti)

        side_eye, bscan_num, sex, age, delta_h = (case_data['side_eye'].item(), 
                                         case_data['BScan'].item(), 
                                         case_data['sex'].item(), 
                                         case_data['age'].item(),
                                         case_data['delta_h'].item())

        target = case_data['label'].item()

        # return bscan_ti, bscan_tj, side_eye, bscan_num, sex, age, delta_h, target, localizer_ti
        return bscan_ti, bscan_tj, side_eye, bscan_num, sex, age, delta_h, target