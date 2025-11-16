"""
ArUco 마커 생성 스크립트 (한글 경로 지원)
광고판에 부착할 마커를 생성합니다.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Windows 콘솔 인코딩 문제 해결
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


def save_image_korean_path(filepath, image):
    """
    한글 경로를 지원하는 이미지 저장 함수
    cv2.imwrite는 한글 경로를 처리하지 못하므로 cv2.imencode 사용
    """
    try:
        # 이미지를 인코딩
        ext = os.path.splitext(filepath)[1]
        result, encoded_img = cv2.imencode(ext, image)

        if result:
            # 파일로 저장
            with open(filepath, 'wb') as f:
                f.write(encoded_img)
            return True
        return False
    except Exception as e:
        print(f"저장 오류: {e}")
        return False


def generate_aruco_markers(output_dir="assets/markers", marker_size=200, num_markers=10):
    """
    ArUco 마커 생성

    Args:
        output_dir: 마커 이미지 저장 경로
        marker_size: 마커 크기 (픽셀)
        num_markers: 생성할 마커 개수
    """
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ArUco 딕셔너리 로드
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    print(f"ArUco 마커 생성 시작... (총 {num_markers}개)")
    print(f"딕셔너리: DICT_4X4_50")
    print(f"마커 크기: {marker_size}x{marker_size} 픽셀")
    print(f"저장 경로: {output_path.absolute()}\n")

    success_count = 0

    for marker_id in range(num_markers):
        try:
            # 마커 이미지 생성
            marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

            # 여백 추가 (인쇄/인식 안정성 향상)
            border_size = 20
            bordered_image = cv2.copyMakeBorder(
                marker_image,
                border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT,
                value=255
            )

            # 파일 저장 (한글 경로 지원)
            filename = output_path / f"marker_{marker_id}.png"
            success = save_image_korean_path(str(filename), bordered_image)

            if success:
                print(f"[OK] 마커 ID {marker_id:2d} 생성 완료 -> {filename.name}")
                success_count += 1
            else:
                print(f"[FAIL] 마커 ID {marker_id:2d} 저장 실패!")

        except Exception as e:
            print(f"[ERROR] 마커 ID {marker_id:2d} 생성 중 오류: {e}")

    # 생성된 파일 확인
    saved_files = list(output_path.glob("marker_*.png"))
    print(f"\n실제 저장된 파일: {len(saved_files)}개")

    if len(saved_files) > 0:
        print("\n생성된 파일 목록:")
        for f in sorted(saved_files):
            file_size = f.stat().st_size
            print(f"  - {f.name} ({file_size} bytes)")

    print(f"\n총 {success_count}/{num_markers}개 마커 생성 완료!")

    if success_count > 0:
        print("\n사용 방법:")
        print("1. assets/markers/ 폴더의 이미지를 A4 용지에 인쇄")
        print("2. 광고판 네 모서리에 마커 부착 (ID 0, 1, 2, 3 권장)")
        print("3. 카메라로 촬영하여 인식 테스트")

        # 4개 마커 레이아웃 가이드 생성
        create_marker_layout_guide(str(output_path), aruco_dict)
    else:
        print("\n[경고] 마커 파일이 생성되지 않았습니다!")


def create_marker_layout_guide(output_dir, aruco_dict, marker_size=150):
    """
    광고판 설치용 4개 마커 레이아웃 가이드 생성
    """
    try:
        # A4 크기 캔버스 (300 DPI 기준)
        canvas_width = 2480
        canvas_height = 3508
        canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

        # 마커 위치 (네 모서리)
        margin = 200
        positions = [
            (margin, margin),  # 좌상단 - ID 0
            (canvas_width - margin - marker_size, margin),  # 우상단 - ID 1
            (canvas_width - margin - marker_size, canvas_height - margin - marker_size),  # 우하단 - ID 2
            (margin, canvas_height - margin - marker_size)  # 좌하단 - ID 3
        ]

        # 4개 마커 배치
        for i, (x, y) in enumerate(positions):
            marker = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
            canvas[y:y+marker_size, x:x+marker_size] = marker

            # 마커 ID 레이블 추가
            label = f"ID: {i}"
            label_y = y + marker_size + 40
            cv2.putText(canvas, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

        # 중앙에 안내 텍스트
        center_y = canvas_height // 2
        instructions = [
            "Smart Ad AR System",
            "Marker Layout Guide",
            "",
            "1. Print this page",
            "2. Attach to advertisement corners",
            "3. Use smartphone camera to scan"
        ]

        for i, text in enumerate(instructions):
            y_pos = center_y + i * 60 - 150
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            x_pos = (canvas_width - text_size[0]) // 2
            cv2.putText(canvas, text, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)

        # 저장 (한글 경로 지원)
        layout_path = Path(output_dir) / "marker_layout_guide.png"
        success = save_image_korean_path(str(layout_path), canvas)

        if success:
            file_size = layout_path.stat().st_size
            print(f"\n[OK] 마커 레이아웃 가이드 생성: {layout_path.name} ({file_size} bytes)")
            print("     -> 이 파일을 인쇄하여 광고판에 부착하세요!")
        else:
            print(f"\n[FAIL] 레이아웃 가이드 저장 실패!")

    except Exception as e:
        print(f"\n[ERROR] 레이아웃 가이드 생성 중 오류: {e}")


if __name__ == "__main__":
    # 현재 스크립트의 상위 디렉토리를 기준으로 경로 설정
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_directory = project_root / "assets" / "markers"

    print(f"OpenCV 버전: {cv2.__version__}")
    print(f"Python 버전: {sys.version.split()[0]}\n")

    generate_aruco_markers(
        output_dir=str(output_directory),
        marker_size=200,
        num_markers=10
    )

    print("\n완료! assets/markers 폴더를 확인하세요.")
