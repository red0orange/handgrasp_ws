import os
from os.path import join as opj
from torch.utils.data import Dataset
import pickle as pkl
from scipy.spatial.transform import Rotation as R


class _3DAPDataset(Dataset):
    """_summary_
    This class is for the data loading.
    """
    def __init__(self, data_dir, mode):
        """_summary_

        Args:
            data_dir (str): path to the dataset
        """
        super().__init__()
        if mode == "train":
            self.data_dir = os.path.join(data_dir, "train")
        elif mode == "test":
            self.data_dir = os.path.join(data_dir, "test")
        else:
            raise ValueError("Mode must be train or test!")

        self.load_data()

    def load_data(self):
        self.all_data = []
        
        shape_ids, pose_data = [], []
        pose_dir = opj(self.data_dir, 'pose_united')
        for file in os.listdir(pose_dir):
            file_dir = opj(pose_dir, file)
            if os.path.isfile(file_dir):
                shape_ids.append(os.path.splitext(file)[0])
                with open(file_dir, 'rb') as f:
                    pose_data.append(pkl.load(f))
        id_poses_dict = dict(zip(shape_ids, pose_data))
        
        with open(opj(self.data_dir, 'full_shape.pkl'), 'rb') as f:
            shape_data = pkl.load(f)
        id_shape_dict = {shape['shape_id']: shape for shape in shape_data}
        
        for id in id_poses_dict.keys():
            for affordance in id_poses_dict[id].keys():
                for pose in id_poses_dict[id][affordance]:
                    new_data_dict = {}
                    new_data_dict['shape_id'] = id
                    new_data_dict['semantic class'] = id_shape_dict[id]['semantic class']
                    new_data_dict['coordinate'] = id_shape_dict[id]['full_shape']['coordinate']
                    new_data_dict['affordance'] = affordance
                    new_data_dict['affordance_label'] = id_shape_dict[id]['full_shape']['label'][affordance]
                    new_data_dict['rotation'] = R.from_matrix(pose[1][:3, :3]).as_quat()
                    new_data_dict['translation'] = pose[1][:3, 3]
                    self.all_data.append(new_data_dict)
            
    def __getitem__(self, index):
        """_summary_

        Args:
            index (int): the element index

        Returns:
            shape id, semantic class, coordinate, affordance text, affordance label, rotation and translation
        """
        data_dict = self.all_data[index]
        return data_dict['shape_id'], data_dict['semantic class'], data_dict['coordinate'], data_dict['affordance'], \
            data_dict['affordance_label'], data_dict['rotation'], data_dict['translation']
        
    def __len__(self):
        return len(self.all_data)