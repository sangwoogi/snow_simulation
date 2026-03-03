import os
import glob
import numpy as np
import open3d as o3d

# 1. 입력 및 출력 경로 설정
input_dir = "data/nuScenes_snow_sev5/samples/LiDAR_TOP"
output_dir = "data/nuScenes_snow_sev5/samples/LiDAR_PLY"

# 출력 폴더가 존재하지 않으면 생성
os.makedirs(output_dir, exist_ok=True)

# 2. 폴더 내의 모든 pcd.bin 파일 목록 가져오기
bin_paths = glob.glob(os.path.join(input_dir, "*.pcd.bin"))
total_files = len(bin_paths)

print(f"총 {total_files}개의 파일을 변환합니다. 잠시만 기다려주세요...\n")

# 3. 각 파일에 대해 변환 작업 수행
for idx, bin_path in enumerate(bin_paths, 1):
    # 파일 이름 추출 및 확장자 변경 (pcd.bin -> ply)
    filename = os.path.basename(bin_path)
    ply_filename = filename.replace(".pcd.bin", ".ply")
    ply_path = os.path.join(output_dir, ply_filename)

    # 데이터 로드
    points = np.fromfile(bin_path, dtype=np.float32)

    # N x 5 reshape (nuScenes: x, y, z, intensity, ring index)
    points = points.reshape(-1, 5)

    xyz = points[:, :3]
    intensity = points[:, 3]

    # intensity를 grayscale로 변환 (0으로 나누는 오류 방지)
    denominator = intensity.max() - intensity.min()
    if denominator == 0:
        intensity_normalized = np.zeros_like(intensity)
    else:
        intensity_normalized = (intensity - intensity.min()) / denominator
        
    colors = np.stack([intensity_normalized]*3, axis=1)

    # Open3D 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ply 저장
    o3d.io.write_point_cloud(ply_path, pcd)
    
    # 진행 상황 출력
    if idx % 50 == 0 or idx == total_files:
        print(f"[{idx}/{total_files}] 변환 완료: {ply_filename}")

print("\n모든 라이다 데이터의 PLY 변환이 완료되었습니다.")