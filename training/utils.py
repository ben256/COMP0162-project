import os
from glob import glob


def create_training_folder(output_dir='../output'):
    tuning_folders = glob(f'{output_dir}/training_*')
    folder_num = [int(x.split('_')[-1]) for x in tuning_folders]
    if len(folder_num) > 0:
        count = max(folder_num) + 1
    else:
        count = 0
    folder_path = f'{output_dir}/training_{count}/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path

    else:
        raise FileExistsError