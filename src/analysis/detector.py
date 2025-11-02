from ultralytics import YOLO
from config import settings

class PersonDetector:
    def __init__(self):
        # [TODO] YOLO 모델 로드 (YOLOv8 예시)
        self.model = YOLO(settings.YOLO_MODEL_PATH)

    def detect_persons(self, frame):
        """
        프레임에서 '사람' 클래스만 감지합니다.
        
        Args:
            frame (np.array): OpenCV BGR 프레임

        Returns:
            list: 감지된 사람들의 바운딩 박스 (x1, y1, x2, y2) 리스트
        """
        # [TODO] 모델 추론 수행
        results = self.model(frame, classes=[0], verbose=False) # class 0 = 'person'
        
        person_boxes = []
        for box in results[0].boxes:
            person_boxes.append(box.xyxy[0].cpu().numpy().astype(int))
            
        return person_boxes