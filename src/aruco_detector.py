"""
ArUco 마커 검출 모듈
실시간으로 카메라 프레임에서 ArUco 마커를 검출하고 광고 영역을 추출합니다.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict


class ArucoDetector:
    """ArUco 마커 검출 및 광고 영역 추출"""

    def __init__(self, dictionary_type=cv2.aruco.DICT_4X4_50):
        """
        초기화

        Args:
            dictionary_type: ArUco 딕셔너리 타입
        """
        # ArUco 딕셔너리 및 파라미터 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 검출 안정화를 위한 변수
        self.last_corners = None
        self.last_ids = None
        self.stable_frame_count = 0
        self.stability_threshold = 3  # 3프레임 연속 검출 시 안정화

        print("✓ ArucoDetector 초기화 완료")

    def detect_markers(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        프레임에서 ArUco 마커 검출

        Args:
            frame: 입력 이미지 프레임

        Returns:
            (corners, ids): 마커 코너 좌표와 ID 배열
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 마커 검출
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        # 검출 결과 반환
        if ids is not None and len(ids) > 0:
            return corners, ids
        else:
            return None, None

    def draw_markers(self, frame: np.ndarray, corners: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """
        검출된 마커를 프레임에 시각화

        Args:
            frame: 원본 프레임
            corners: 마커 코너 좌표
            ids: 마커 ID

        Returns:
            마커가 그려진 프레임
        """
        if corners is None or ids is None:
            return frame

        # 마커 외곽선 그리기
        frame_with_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

        # 각 마커의 중심에 ID 표시
        for i, corner in enumerate(corners):
            # 마커 중심 계산
            center = corner[0].mean(axis=0).astype(int)

            # ID 텍스트 추가
            cv2.putText(
                frame_with_markers,
                f"ID:{ids[i][0]}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame_with_markers

    def extract_roi(self, frame: np.ndarray, corners: np.ndarray, ids: np.ndarray,
                    target_ids: List[int] = [0, 1, 2, 3]) -> Optional[np.ndarray]:
        """
        4개의 마커로 둘러싸인 영역(ROI) 추출

        Args:
            frame: 원본 프레임
            corners: 마커 코너 좌표
            ids: 마커 ID
            target_ids: 추출할 영역을 정의하는 4개 마커 ID (좌상, 우상, 우하, 좌하 순서)

        Returns:
            추출된 ROI 이미지 (정면 뷰로 변환됨)
        """
        if corners is None or ids is None or len(ids) < 4:
            return None

        # 필요한 4개 마커가 모두 검출되었는지 확인
        marker_dict = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in target_ids:
                marker_dict[marker_id] = corners[i][0]

        if len(marker_dict) < 4:
            return None

        # 마커 순서대로 코너 포인트 추출 (각 마커의 내측 코너)
        # 좌상(ID 0): 우하 코너, 우상(ID 1): 좌하 코너,
        # 우하(ID 2): 좌상 코너, 좌하(ID 3): 우상 코너
        src_points = np.array([
            marker_dict[target_ids[0]][2],  # 좌상 마커의 우하 코너
            marker_dict[target_ids[1]][3],  # 우상 마커의 좌하 코너
            marker_dict[target_ids[2]][0],  # 우하 마커의 좌상 코너
            marker_dict[target_ids[3]][1]   # 좌하 마커의 우상 코너
        ], dtype=np.float32)

        # ROI 크기 계산 (마커 간 거리 기반)
        width = int(max(
            np.linalg.norm(src_points[0] - src_points[1]),
            np.linalg.norm(src_points[2] - src_points[3])
        ))
        height = int(max(
            np.linalg.norm(src_points[0] - src_points[3]),
            np.linalg.norm(src_points[1] - src_points[2])
        ))

        # 목표 좌표 (정면 뷰)
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # Perspective Transform 행렬 계산
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # ROI 추출
        roi = cv2.warpPerspective(frame, matrix, (width, height))

        return roi

    def get_homography_matrix(self, corners: np.ndarray, ids: np.ndarray,
                               target_ids: List[int] = [0, 1, 2, 3],
                               roi_size: Tuple[int, int] = (800, 600)) -> Optional[np.ndarray]:
        """
        오버레이를 위한 Homography 역변환 행렬 계산

        Args:
            corners: 마커 코너 좌표
            ids: 마커 ID
            target_ids: 4개 마커 ID
            roi_size: ROI 크기 (width, height)

        Returns:
            Homography 역변환 행렬 (ROI → 원본 프레임)
        """
        if corners is None or ids is None or len(ids) < 4:
            return None

        # 마커 딕셔너리 생성
        marker_dict = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in target_ids:
                marker_dict[marker_id] = corners[i][0]

        if len(marker_dict) < 4:
            return None

        # 원본 프레임의 마커 코너 포인트
        dst_points = np.array([
            marker_dict[target_ids[0]][2],  # 좌상
            marker_dict[target_ids[1]][3],  # 우상
            marker_dict[target_ids[2]][0],  # 우하
            marker_dict[target_ids[3]][1]   # 좌하
        ], dtype=np.float32)

        # ROI 좌표계의 코너 포인트
        src_points = np.array([
            [0, 0],
            [roi_size[0] - 1, 0],
            [roi_size[0] - 1, roi_size[1] - 1],
            [0, roi_size[1] - 1]
        ], dtype=np.float32)

        # Homography 역변환 행렬 (ROI → 원본)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        return matrix

    def get_detection_info(self, corners: np.ndarray, ids: np.ndarray) -> Dict:
        """
        검출된 마커 정보 요약

        Args:
            corners: 마커 코너 좌표
            ids: 마커 ID

        Returns:
            마커 정보 딕셔너리
        """
        if corners is None or ids is None:
            return {
                "detected": False,
                "count": 0,
                "ids": []
            }

        info = {
            "detected": True,
            "count": len(ids),
            "ids": ids.flatten().tolist(),
            "centers": []
        }

        # 각 마커의 중심 좌표 계산
        for corner in corners:
            center = corner[0].mean(axis=0).tolist()
            info["centers"].append(center)

        return info
