"""절대 경로로 마커 생성"""
import cv2
import numpy as np
import os

# 절대 경로 직접 지정
output_dir = r"C:\Users\송현\Documents\2025\2025 캡디\capstone\assets\markers"

# 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

print(f"저장 경로: {output_dir}")
print(f"OpenCV: {cv2.__version__}\n")

# ArUco 딕셔너리
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 마커 생성
for i in range(10):
    marker = cv2.aruco.generateImageMarker(aruco_dict, i, 200)
    bordered = cv2.copyMakeBorder(marker, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    filename = os.path.join(output_dir, f"marker_{i}.png")
    result = cv2.imwrite(filename, bordered)

    print(f"{'[OK]' if result else '[FAIL]'} ID {i}: {os.path.basename(filename)}")

# 확인
import glob
files = glob.glob(os.path.join(output_dir, "marker_*.png"))
print(f"\n총 {len(files)}개 파일 생성됨")

# 레이아웃 가이드
canvas = np.ones((3508, 2480), dtype=np.uint8) * 255
for i, (x, y) in enumerate([(200,200), (2130,200), (2130,3158), (200,3158)]):
    m = cv2.aruco.generateImageMarker(aruco_dict, i, 150)
    canvas[y:y+150, x:x+150] = m

layout_file = os.path.join(output_dir, "marker_layout_guide.png")
cv2.imwrite(layout_file, canvas)
print(f"레이아웃 가이드: {os.path.basename(layout_file)}")
