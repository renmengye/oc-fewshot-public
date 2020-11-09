import os
import sys
import glob
import numpy as np
import multiprocessing
from zipfile import ZipFile
import argparse
import shutil

_FILES_TO_UNZIP = [
    'house_segmentations',
    'matterport_camera_intrinsics',
    'matterport_skybox_images',
    'region_segmentations',
    'sens',
    'undistorted_camera_parameters',
    'undistorted_color_images',
    'undistorted_depth_images'
]

def split_processes(folders, num_processes):
    folders = np.array(folders)
    num_per_process = len(folders) // num_processes
    num_per_process_r = len(folders) % num_processes

    idxs = list(range(len(folders)))

    split_folders = []
    for proc_i in range(num_processes):
        start_i = proc_i*num_per_process

        if proc_i == num_processes-1:
            end_i = None
        else:
            end_i = (proc_i+1)*num_per_process
        
        idx_i = np.array(idxs[start_i:end_i])
        folder_i = folders[idx_i]
        split_folders.append(folder_i)
    return split_folders

def process(folders):
    for folder in folders:
        for unzip_i, file_to_unzip in enumerate(_FILES_TO_UNZIP):
            temp_folder = os.path.join(folder, os.path.basename(folder))
            if os.path.exists(temp_folder):
                print(f"Removing temp folder: {temp_folder}")
                shutil.rmtree(temp_folder)
            
            final_zip_foldername = os.path.join(folder, file_to_unzip)
            zip_path = f'{final_zip_foldername}.zip'
            if os.path.exists(final_zip_foldername):
                print(f"\n{final_zip_foldername}")
                continue
            if not os.path.exists(zip_path):
                print(f"\n{zip_path} DNE")
                continue
            
            print(f"Extracting {zip_path}...")
            with ZipFile(zip_path, 'r') as zipObj:
                zipObj.extractall(folder)

            outpath = os.path.join(folder, os.path.basename(folder))
            try:
                out_list = os.listdir(outpath)
                assert len(out_list) == 1, 'ERROR'
                unzipped_dir = os.path.join(outpath, out_list[0])
                shutil.move(unzipped_dir, folder)
                shutil.rmtree(outpath)
            except Exception as e:
                print(f"\nEXCEPTION: {e}")
    
def main(args):
    folders = glob.glob(f'{args.root}/*')
    
    split_folders = split_processes(folders, args.n_processes)
    jobs = []
    for proc_i, folders_i in enumerate(split_folders):
        p = multiprocessing.Process(target=process, args=(folders_i,))
        jobs.append(p)
        p.start()
        
    print("\nFin.")

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='root', type=str,
                        default='/hdd/mliuzzolino/Matterport/v1/scans',
                        help='Directory of scans; e.g., data/v1/scans')
    parser.add_argument('-n', dest='n_processes', type=int,
                        default=5,
                        help='Number of subprocesses')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = setup_args()
    main(args)