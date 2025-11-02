import cv2

# 시각화용 색상 및 폰트 설정 (BGR)
BOX_COLOR = (0, 255, 0)   # Green
TEXT_COLOR = (0, 0, 0)    # Black
BG_COLOR = (0, 255, 0)    # Green (텍스트 배경)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

def draw_results(frame, boxes, labels):
    """
    OpenCV 프레임에 바운딩 박스와 레이블을 그립니다.

    Args:
        frame (np.array): 원본 BGR 프레임
        boxes (list): (x1, y1, x2, y2) 좌표 리스트
        labels (list): e.g., ["20s_female", "40s_male"]

    Returns:
        np.array: 박스와 레이블이 그려진 프레임
    """
    # 원본 프레임을 수정하지 않기 위해 복사
    output_frame = frame.copy()

    if len(boxes) != len(labels):
        print("Warning: Mismatch between number of boxes and labels.")
        return frame # 오류 시 원본 반환

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box

        # 1. 바운딩 박스 그리기
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), BOX_COLOR, FONT_THICKNESS)

        # 2. 텍스트 레이블 배경 그리기
        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        
        # 텍스트 배경 박스 좌표 계산 (박스 위에 위치)
        text_bg_y1 = max(y1 - text_height - baseline, 0) # 프레임 상단 밖으로 나가지 않도록
        text_bg_y2 = y1 - baseline
        
        cv2.rectangle(output_frame, (x1, text_bg_y1), (x1 + text_width, text_bg_y2), BG_COLOR, -1) # -1: 채우기

        # 3. 텍스트 그리기
        cv2.putText(output_frame, label, (x1, y1 - baseline - 5), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

    return output_frame