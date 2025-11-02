import torch
import torchvision.transforms as transforms
# [TODO] 실제 사용할 모델 아키텍처 임포트 (e.g., from .models import MobileNetV2)

class DemographicClassifier:
    def __init__(self):
        # [TODO] PyTorch 모델 로드 및 .eval() 모드 설정
        self.model = None # [TODO]
        # self.model = torch.load(settings.AGE_GENDER_MODEL_PATH)
        # self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify_demographics(self, frame, person_boxes):
        """
        감지된 사람들 영역을 잘라내어 연령/성별을 분류합니다.

        Args:
            frame (np.array): 원본 BGR 프레임
            person_boxes (list): (x1, y1, x2, y2) 리스트

        Returns:
            list: 각 사람의 예측 결과 (e.g., "20s_female", "40s_male") 리스트
        """
        results = []
        if not person_boxes:
            return results

        for box in person_boxes:
            x1, y1, x2, y2 = box
            # [TODO] 사람 영역 자르기(crop) 및 전처리
            person_crop = frame[y1:y2, x1:x2]
            # input_tensor = self.transform(person_crop).unsqueeze(0)
            
            # [TODO] 모델 추론 (연령/성별 예측)
            # with torch.no_grad():
            #     age_pred, gender_pred = self.model(input_tensor)
            
            # [TODO] 예측 결과를 정해진 카테고리(e.g., "20s_female")로 변환
            age_category = "20s"
            gender_category = "female"
            
            results.append(f"{age_category}_{gender_category}")
            
        return results