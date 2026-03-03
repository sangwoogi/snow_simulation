import os
import cv2
import numpy as np
import argparse
import shutil
from glob import glob
from tqdm import tqdm

# LiDAR_corruptions에서 KITTI용 일반 snow_sim 함수를 가져옵니다.
from LiDAR_corruptions import snow_sim
from Camera_corruptions import ImageAddSnow

def parse_args():
    parser = argparse.ArgumentParser(description='Apply snow corruptions to KITTI-360 dataset offline.')
    parser.add_argument('--dataroot', type=str, default='data/KITTI-360', help='Path to raw KITTI-360 dataset')
    parser.add_argument('--severity', type=int, default=1, choices=[0, 1, 2, 3, 4, 5], help='Corruption severity')
    return parser.parse_args()

def process_offline(args):
    dataroot = args.dataroot
    severity = args.severity
    
    out_dir = f"{dataroot}_snow_sev{severity}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Data will be saved to: {out_dir}")

    if severity > 0:
        snow_sim_cam = ImageAddSnow(severity=severity, seed=2026)

    '''
    # 1. 2D 이미지 처리 (data_2d_raw)
    image_files = glob(os.path.join(dataroot, 'data_2d_raw', '**', '*.png'), recursive=True)
    for src_path in tqdm(image_files, desc=f"Processing KITTI-360 Images (Severity {severity})"):
        rel_path = os.path.relpath(src_path, dataroot)
        dst_path = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if severity == 0:
            shutil.copy2(src_path, dst_path)
        else:
            img_bgr = cv2.imread(src_path)
            if img_bgr is not None:
                img_rgb = img_bgr[:, :, [2, 1, 0]]
                img_aug_rgb = snow_sim_cam(image=img_rgb)
                img_aug_bgr = img_aug_rgb[:, :, [2, 1, 0]]
                cv2.imwrite(dst_path, img_aug_bgr)
    '''
    
    # 2. 3D LiDAR 처리 (data_3d_raw)
    target_drives = [
        '2013_05_28_drive_0000_sync',
        '2013_05_28_drive_0002_sync',
        '2013_05_28_drive_0003_sync'
    ]
    
    lidar_files = []
    for drive in target_drives:
        # 지정된 드라이브 폴더 내의 bin 파일만 탐색합니다.
        search_path = os.path.join(dataroot, 'data_3d_raw', drive, '**', '*.bin')
        lidar_files.extend(glob(search_path, recursive=True))

    for src_path in tqdm(lidar_files, desc=f"Processing KITTI-360 LiDAR (Severity {severity})"):
        rel_path = os.path.relpath(src_path, dataroot)
        dst_path = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if severity == 0:
            shutil.copy2(src_path, dst_path)
        else:
            # KITTI-360 LiDAR는 일반적으로 (N, 4) 차원입니다 (x, y, z, reflectance)
            pc = np.fromfile(src_path, dtype=np.float32).reshape(-1, 4)
            
            # 함수 내부에서 요구하는 입력 형식에 맞게 x, y, z 좌표만 넘길지 등 확인이 필요합니다.
            pc_aug = snow_sim(pointcloud=pc, severity=severity)
            
            pc_aug.astype(np.float32).tofile(dst_path)

    # 3. 그 외 필수 폴더 및 파일 복사 (calibration, data_poses 등)
    print("[*] Copying calibration and pose data...")
    for dir_name in ['calibration', 'data_poses']:
        src_dir = os.path.join(dataroot, dir_name)
        dst_dir = os.path.join(out_dir, dir_name)
        if os.path.exists(src_dir) and not os.path.exists(dst_dir):
            shutil.copytree(src_dir, dst_dir)

    print("[*] Offline corruption processing complete!")

if __name__ == '__main__':
    args = parse_args()
    process_offline(args)