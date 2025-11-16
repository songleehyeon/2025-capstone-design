"""간단한 ArUco 마커 생성 스크립트"""
import cv2
import numpy as np
from pathlib import Path

# 경로 설정
output_dir = Path(__file__).parent.parent / "assets" / "markers"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"저장 경로: {output_dir.absolute()}")
print(f"OpenCV 버전: {cv2.__version__}\n")

# ArUco 딕셔너리 로드
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 10개 마커 생성
for marker_id in range(10):
    # 마커 생성
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)

    # 여백 추가
    bordered = cv2.copyMakeBorder(marker_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    # 저장
    filepath = output_dir / f"marker_{marker_id}.png"
    success = cv2.imwrite(str(filepath), bordered)

    print(f"{'[OK]' if success else '[FAIL]'} 마커 {marker_id}: {filepath.name}")

# 파일 확인
saved_files = list(output_dir.glob("marker_*.png"))
print(f"\n생성된 파일: {len(saved_files)}개")

# 레이아웃 가이드 생성
canvas = np.ones((3508, 2480), dtype=np.uint8) * 255
positions = [(200, 200), (2130, 200), (2130, 3158), (200, 3158)]

for i, (x, y) in enumerate(positions):
    marker = cv2.aruco.generateImageMarker(aruco_dict, i, 150)
    canvas[y:y+150, x:x+150] = marker
    cv2.putText(canvas, f"ID: {i}", (x, y+190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

layout_path = output_dir / "marker_layout_guide.png"
cv2.imwrite(str(layout_path), canvas)
print(f"[OK] 레이아웃 가이드: {layout_path.name}")
