import os
import cv2
import numpy as np
import argparse
import shutil
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

from LiDAR_corruptions import snow_sim_nus
from Camera_corruptions import ImageAddSnow

def parse_args():
    parser = argparse.ArgumentParser(description='Apply snow corruptions to nuScenes dataset offline.')
    parser.add_argument('--dataroot', type=str, default='data/nuScenes', help='Path to raw nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='nuScenes version (e.g., v1.0-mini, v1.0-trainval)')
    parser.add_argument('--severity', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help='Corruption severity (0=No corruption, 1-5=Snow intensity)')
    return parser.parse_args()

def process_offline(args):
    dataroot = args.dataroot
    severity = args.severity
    
    # 변환된 데이터를 저장할 새로운 루트 디렉토리 설정
    out_dir = f"{dataroot}_snow_sev{severity}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Data will be saved to: {out_dir}")

    # nuScenes 인스턴스 초기화
    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=True)

    # 카메라 모듈 초기화 (severity 0이 아닐 때만)
    if severity > 0:
        snow_sim_cam = ImageAddSnow(severity=severity, seed=2026) #

    # 모든 샘플 순회
    for sample in tqdm(nusc.sample, desc=f"Processing nuScenes (Severity {severity})"):
        
        # 1. 카메라 센서 처리
        cam_sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                       'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for cam in cam_sensors:
            cam_data = nusc.get('sample_data', sample['data'][cam])
            rel_path = cam_data['filename']
            src_path = os.path.join(dataroot, rel_path)
            dst_path = os.path.join(out_dir, rel_path)
            
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            if severity == 0:
                shutil.copy2(src_path, dst_path)
            else:
                # mmdet3d와 동일한 채널 순서 맞추기 (BGR -> RGB 변환)
                img_bgr = cv2.imread(src_path)
                if img_bgr is not None:
                    img_rgb = img_bgr[:, :, [2, 1, 0]] #
                    img_aug_rgb = snow_sim_cam(image=img_rgb) #
                    img_aug_bgr = img_aug_rgb[:, :, [2, 1, 0]] #
                    cv2.imwrite(dst_path, img_aug_bgr) #

        # 2. LiDAR 센서 처리
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        rel_path = lidar_data['filename']
        src_path = os.path.join(dataroot, rel_path)
        dst_path = os.path.join(out_dir, rel_path)
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if severity == 0:
            shutil.copy2(src_path, dst_path)
        else:
            # nuScenes 포인트 클라우드는 float32이며 (N, 5) 차원을 가짐
            pc = np.fromfile(src_path, dtype=np.float32).reshape(-1, 5) 
            
            # LiDAR snow 시뮬레이션 적용
            pc_aug = snow_sim_nus(pointcloud=pc, severity=severity) #
            
            # 다시 .bin 파일로 저장
            pc_aug.astype(np.float32).tofile(dst_path)

    print("[*] Offline corruption processing complete!")

if __name__ == '__main__':
    args = parse_args()
    process_offline(args)