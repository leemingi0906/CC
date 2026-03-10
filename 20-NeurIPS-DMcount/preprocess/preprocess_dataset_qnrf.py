import os
import glob
import cv2
import numpy as np
import scipy.io as sio
from PIL import Image

# =========================================================
# [설정] 원본 데이터 경로 자동 탐색
# 설명: 사용자의 실행 위치가 달라도 데이터를 찾을 수 있도록 여러 후보 경로를 검사합니다.
# =========================================================
CANDIDATE_PATHS = [
    './UCF/UCF-QNRF',           # 1. 현재 폴더 아래 UCF/UCF-QNRF
    '../UCF/UCF-QNRF',          # 2. 상위 폴더 아래 UCF/UCF-QNRF (preprocess 폴더 안에서 실행 시 유력)
    './UCF/UCF-QNRF_ECCV18',    # 3. 폴더명이 ECCV18인 경우
    '../UCF/UCF-QNRF_ECCV18',   # 4. 상위 폴더 + ECCV18
    './UCF-QNRF',               # 5. 현재 폴더 바로 아래
    '../UCF-QNRF',              # 6. 상위 폴더 바로 아래
    '/home/kimsooyeon/Desktop/CC/20-NeurIPS-DMcount/UCF/UCF-QNRF' # 7. 절대 경로 예시
]

# 전처리된 데이터가 저장될 경로
OUTPUT_DATASET_PATH = './UCF-QNRF_Processed'

# 폴더 매핑 (원본폴더명 -> 저장폴더명)
FOLDER_MAPPING = {
    'Train': 'train',
    'Test':  'test'
}
# =========================================================

def find_dataset_path(candidates):
    print("🔍 데이터셋 경로를 찾는 중...")
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            # 경로가 존재하더라도 안에 Train 폴더가 있는지 확인
            if os.path.exists(os.path.join(abs_path, 'Train')) or os.path.exists(os.path.join(abs_path, 'train')):
                return abs_path
    return None

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def generate_data(im_path, min_size, max_size):
    try:
        im = Image.open(im_path).convert('RGB')
    except Exception as e:
        print(f"   ❌ Error opening image: {im_path}")
        return None, None

    im_w, im_h = im.size
    
    # GT 파일 경로 (예: img_0001.jpg -> img_0001_ann.mat)
    mat_path = im_path.replace('.jpg', '_ann.mat')
    
    # 대소문자 문제 처리 (Linux는 대소문자 구분)
    if not os.path.exists(mat_path):
        # 확장자가 .JPG인 경우 .mat은 그대로 소문자인지 체크 등 예외 처리
        if not os.path.exists(mat_path):
             print(f"   ⚠️ Annotation not found: {mat_path}")
             return None, None

    try:
        points = sio.loadmat(mat_path)['annPoints'].astype(np.float32)
    except Exception as e:
        print(f"   ❌ Error loading mat: {mat_path}")
        return None, None
    
    # 이미지 범위 밖 포인트 제거
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    
    # 리사이징 계산
    new_h, new_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    
    if rr != 1.0:
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        points = points * rr
        
    return Image.fromarray(im), points

def main():
    min_size = 512
    max_size = 2048

    # 1. 데이터셋 경로 찾기
    origin_path = find_dataset_path(CANDIDATE_PATHS)
    
    if origin_path is None:
        print("\n❌ [Error] 데이터셋 경로를 찾을 수 없습니다.")
        print("   다음 경로들을 확인해보았으나 실패했습니다:")
        for p in CANDIDATE_PATHS:
            print(f"   - {os.path.abspath(p)}")
        print("\n👉 해결방법: 'CANDIDATE_PATHS' 리스트에 실제 데이터셋 경로를 추가하거나, 터미널 위치를 확인하세요.")
        return

    print(f"✅ 데이터셋 경로 확인됨: {origin_path}")
    print(f"📂 결과 저장 경로: {os.path.abspath(OUTPUT_DATASET_PATH)}")

    for src_folder_name, target_folder_name in FOLDER_MAPPING.items():
        src_path_full = os.path.join(origin_path, src_folder_name)
        save_path_full = os.path.join(OUTPUT_DATASET_PATH, target_folder_name)
        
        print(f"\n========================================")
        print(f"🚀 Processing: {src_folder_name} -> {target_folder_name}")
        print(f"   Source: {src_path_full}")
        print(f"========================================")

        if not os.path.exists(src_path_full):
            # 혹시 폴더명이 소문자(train/test)일 수도 있으니 한번 더 체크
            src_path_lower = os.path.join(origin_path, src_folder_name.lower())
            if os.path.exists(src_path_lower):
                print(f"   ℹ️ 대문자 폴더({src_folder_name}) 대신 소문자 폴더를 찾았습니다.")
                src_path_full = src_path_lower
            else:
                print(f"⚠️ Warning: Source folder not found: {src_path_full}")
                continue

        # 폴더 내 모든 jpg 파일 검색 (대소문자 무시하지 않으므로 jpg만 검색)
        im_list = sorted(glob.glob(os.path.join(src_path_full, '*.jpg')))
        total_imgs = len(im_list)
        
        print(f"📸 Found {total_imgs} images.")

        if total_imgs == 0:
            print("⚠️ No .jpg images found. Trying .JPG...")
            im_list = sorted(glob.glob(os.path.join(src_path_full, '*.JPG')))
            total_imgs = len(im_list)
            print(f"📸 Found {total_imgs} .JPG images.")

        if total_imgs == 0:
            print("❌ No images found. Skipping this folder.")
            continue

        # 저장 경로 생성
        os.makedirs(save_path_full, exist_ok=True)
        
        count = 0
        for i, im_path in enumerate(im_list):
            filename = os.path.basename(im_path)
            
            # 데이터 생성
            im, points = generate_data(im_path, min_size, max_size)
            
            if im is None: continue

            # 저장 (이미지 & 좌표)
            im_save_path = os.path.join(save_path_full, filename)
            im.save(im_save_path)
            
            gd_save_path = im_save_path.replace('.jpg', '.npy').replace('.JPG', '.npy')
            np.save(gd_save_path, points)
            
            count += 1
            if count % 50 == 0:
                print(f"   Processed {count}/{total_imgs} ...")

    print(f"\n✅ All Done! Processed data saved to: {os.path.abspath(OUTPUT_DATASET_PATH)}")

if __name__ == '__main__':
    main()