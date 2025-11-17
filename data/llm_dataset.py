from torch.utils.data import Dataset
import numpy as np

from .dataset import VibrationDataset

class LLM_Dataset(Dataset):
    def __init__(self, vibration_dataset:VibrationDataset,
                include_ref= True):
        super().__init__()
        
        self.vibration_dataset = vibration_dataset
        
    def __len__(self):
        return len(self.vibration_dataset)

    def feature_extract(self, vibration:np.array):
        """_summary_

        Args:
            vibration (np.array): 진동 데이터

        Returns:
            feature_dict (dict): vibration에서 추출한 특징 dict
        """
        feature_dict = {}
        return feature_dict
    
    def __getitem__(self, index):
        
        data_dict = self.vibration_dataset[index]
        
        x_feat = self.feature_extract(data_dict['x_vib'])
        
        if 'ref_vib' in data_dict.keys():
            ref_feat = self.feature_extract(data_dict['ref_vib'])
        
        prompt = """
            LLM 에 들어갈 입력 Prompt
        """
        
        
        llm_data_dict = {
            'prompt' : prompt
        }
        
        return data_dict.update(llm_data_dict)
    
def get_llm_dataset(train_dataset, val_dataset):
    train_llm_dataset = LLM_Dataset(
        vibration_dataset=train_dataset
    )
    val_llm_dataset = LLM_Dataset(
        vibration_dataset=val_dataset
    )
    return train_llm_dataset, val_llm_dataset