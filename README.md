

# Data folder restructure
The data folder was restructured to make experiments easier. The bash script to restructure the data can be found in `change_data_folder_structure.sh`. The old format was the following:

    data/
    ├── sub-<user_id>/
    │   ├── connectome/
    │   │   ├── sub-<user_id>_ses-01_atlas-Glasser_SC.npy
    │   │   ├── sub-<user_id>_ses-01_atlas-Schaefer1000_SC.npy
    │   │   ├── sub-<user_id>_task-rest_run-1_atlas-Glasser_desc-lrrl_FC.npy
    │   │   ├── sub-<user_id>_task-rest_run-1_atlas-Schaefer1000_desc-lrrl_FC.npy
    │   │   ├── sub-<user_id>_task-rest_run-2_atlas-Glasser_desc-lrrl_FC.npy
    │   │   ├── sub-<user_id>_task-rest_run-2_atlas-Schaefer1000_desc-lrrl_FC.npy


The new format is:

    data_new_struct/
    ├── Glasser/
    │   ├── sub-<user_id>/
    │   │   ├── ses/
    │   │   │   ├── sub-<user_id>_ses-01_atlas-Glasser_SC.npy
    │   │   ├── task-rest/
    │   │   │   ├── sub-<user_id>_task-rest_run-1_atlas-Glasser_desc-lrrl_FC.npy
    ├── Schaefer1000/
    │   ├── sub-<user_id>/
    │   │   ├── ses/
    │   │   │   ├── sub-<user_id>_ses-01_atlas-Schaefer1000_SC.npy
    │   │   ├── task-rest/
    │   │   │   ├── sub-<user_id>_task-rest_run-1_atlas-Schaefer1000_desc-lrrl_FC.npy


# Data processing

## Loading networks

After `change_data_folder_structure.sh` has been called, the data can be loaded using the `NetworkLoader` in `graph_matching/data_loader/graph_loader.py`. An example of use can be found in `graph_loader.py`.


### Average network
You can calculate the average network given a `NetworkLoader` using the `compute_mean_network` found in `graph_matching/data_loader/graph_loader.py`. See the script for example use.
