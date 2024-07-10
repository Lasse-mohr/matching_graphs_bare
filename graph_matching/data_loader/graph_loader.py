""" 
A dataclass to load the networks stored in data_new_struct.

The loader assumes the folder structure created by change_data_folder_struct.sh
"""
import os
import numpy as np

class NetworkLoader:

    VALID_ATLASES = ['Glasser', 'Schaefer1000']
    VALID_STRUCTURE_TYPES = ['ses-01', 'task-rest']

    def __init__(self, atlas, structure_type, seed=1, base_dir='./data_new_struct'):
        if atlas not in self.VALID_ATLASES:
            raise ValueError(f"Invalid atlas: {atlas}. Must be one of {self.VALID_ATLASES}")
        if structure_type not in self.VALID_STRUCTURE_TYPES:
            raise ValueError(f"Invalid structure type: {structure_type}. Must be one of {self.VALID_STRUCTURE_TYPES}")

        self.base_dir = base_dir
        self.atlas = atlas
        self.atlas_coordinates = self._load_atlas_coord_()
        self.structure_type = structure_type
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.file_list = self._collect_files()
        self.index = 0

    def _load_atlas_coord_(self):
        if self.atlas == 'Schaefer1000':
            atlas_path = os.path.join(self.base_dir, 'atlas_coordinates/coordinate_Schaefer2.npy')
        elif self.atlas == 'Glasser':
            atlas_path = os.path.join(self.base_dir, 'atlas_coordinates/coordinates_Gla358.npy')

        return np.load(atlas_path)

    def _collect_files(self):
        """
        Collect the list of network files based on the atlas and structure type.
        """
        file_list = []
        atlas_path = os.path.join(self.base_dir, self.atlas)
        if not os.path.isdir(atlas_path):
            return file_list
        
        for user_dir in os.listdir(atlas_path):
            user_path = os.path.join(atlas_path, user_dir)
            if not os.path.isdir(user_path):
                continue
            
            structure_dir = os.path.join(user_path, self.structure_type)
            if not os.path.isdir(structure_dir):
                continue
            
            for file_name in os.listdir(structure_dir):
                if self.structure_type == 'ses-01' and file_name.endswith('_SC.npy'):
                    file_list.append((user_dir, os.path.join(structure_dir, file_name)))
                elif self.structure_type == 'task-rest' and file_name.endswith('_desc-lrrl_FC.npy'):
                    file_list.append((user_dir, os.path.join(structure_dir, file_name)))

        # Shuffle the list of files for random iteration
        self.rng.shuffle(file_list)
        return file_list

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.file_list):
            user_id, file_path = self.file_list[self.index]
            network = np.load(file_path)
            self.index += 1
            return user_id, network
        else:
            raise StopIteration

def compute_median_network(loader: NetworkLoader):
    # loader is used as an iterator, so we need a copy with the full list of networks
    loader_copy = NetworkLoader(
        atlas=loader.atlas,
        structure_type=loader.structure_type,
        seed=loader.seed,
        base_dir=loader.base_dir,
        )
    array_stack = np.stack([network for _, network in loader_copy], axis=0)
    return np.median(array_stack, axis=0)

def compute_mean_network(loader: NetworkLoader):
    # loader is used as an iterator, so we need a copy with the full list of networks
    loader_copy = NetworkLoader(
        atlas=loader.atlas,
        structure_type=loader.structure_type,
        seed=loader.seed,
        base_dir=loader.base_dir,
        )
    array_stack = np.stack([network for _, network in loader_copy], axis=0)
    return np.mean(array_stack, axis=0)

# Example use
if __name__ == '__main__':
    atlases = ['Glasser', 'Schaefer1000']
    structure_types = ['task-rest', 'ses-01']

    # loading the different type of networks
    for atlas in atlases:
        for structure_type in structure_types:

            loader = NetworkLoader(atlas, structure_type)
            
            print(f'Atlas: {atlas}, structure: {structure_type}')
            for user_id, network in loader:
                print(f"User: {user_id}, Network shape: {network.shape}")
                break
            
            print('')

    # computing the mean network given a loader
    mean_network = compute_mean_network(loader=loader)