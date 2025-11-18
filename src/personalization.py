"""
개인화 로직 모듈
사용자 프로필과 OCR 텍스트를 기반으로 개인화된 콘텐츠를 생성합니다.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from universal_recommendation_engine import UniversalRecommendationEngine


class PersonalizationEngine:
    """개인화 콘텐츠 생성 엔진"""

    def __init__(self, profile_path: str = "config/user_profile.json"):
        """
        초기화

        Args:
            profile_path: 사용자 프로필 JSON 파일 경로
        """
        if not Path(profile_path).is_absolute():
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            profile_path = str(project_root / profile_path)

        self.profile_path = profile_path
        self.user_profile = self._load_profile()

        # 통합 추천 엔진 초기화
        self.recommendation_engine = UniversalRecommendationEngine()

        print("✓ PersonalizationEngine 초기화 완료")
        print(f"  사용자 ID: {self.user_profile.get('user_id', 'Unknown')}")

    def _load_profile(self) -> Dict:
        """사용자 프로필 로드"""
        try:
            with open(self.profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            return profile
        except FileNotFoundError:
            print(f"경고: 프로필 파일을 찾을 수 없습니다: {self.profile_path}")
            return self._get_default_profile()
        except json.JSONDecodeError:
            print(f"경고: 프로필 파일 형식 오류: {self.profile_path}")
            return self._get_default_profile()

    def _get_default_profile(self) -> Dict:
        """기본 프로필 반환"""
        return {
            "user_id": "default",
            "gender": "female",
            "age": 25,
            "age_group": "20s",
            "occupation": [],
            "living_type": [],
            "allergies": [],
            "vegan": False,
            "low_sugar_preference": False,
            "low_caffeine_preference": False,
            "price_sensitivity": "medium",
            "attribute_preferences": [],
            "context": {
                "current_time": "afternoon",
                "day_type": "weekday",
                "weather": "normal"
            },
            "personalization_level": "medium"
        }

    def analyze_text_content(self, ocr_results: List[Dict]) -> Dict:
        """
        OCR 텍스트 내용 분석 (브랜드 감지)

        Args:
            ocr_results: OCR 결과 리스트

        Returns:
            분석 결과 (브랜드 정보)
        """
        all_text = ' '.join([r['text'] for r in ocr_results])

        analysis = {
            'brand': None
        }

        # 통합 룰 엔진을 통한 브랜드 감지
        detected_brand = self.recommendation_engine.detect_brand(all_text)
        if detected_brand:
            analysis['brand'] = detected_brand

        return analysis

    def generate_personalized_content(self, content_analysis: Dict) -> List[Dict]:
        """
        개인화된 콘텐츠 생성

        Args:
            content_analysis: 콘텐츠 분석 결과

        Returns:
            개인화된 오버레이 항목 리스트
        """
        overlays = []

        # 브랜드 감지 시 해당 브랜드의 규칙 기반 추천 시스템 사용
        brand = content_analysis.get('brand')
        if brand:
            print(f"✓ {brand} 브랜드 감지 - 규칙 기반 추천 시스템 적용")
            print(f"  사용자 프로필: {self.user_profile.get('user_id', 'Unknown')}")

            # 통합 추천 엔진을 통한 추천 생성
            recommendations = self.recommendation_engine.get_recommendations(brand, self.user_profile)
            print(f"  생성된 추천 개수: {len(recommendations)}")

            # 추천 결과를 오버레이 형식으로 변환
            for idx, rec in enumerate(recommendations[:5]):  # 최대 5개 표시
                print(f"  추천 {idx+1}: {rec.get('message', 'N/A')} (우선순위: {rec.get('priority', 0)})")

                overlay = {
                    'type': rec.get('type', 'info'),
                    'position': [50, 50 + (idx * 80)],  # 세로로 배치
                    'content': rec.get('message', ''),
                    'color': rec.get('color', (100, 200, 100)),
                    'priority': rec.get('priority', 0),
                    'rule_id': rec.get('rule_id', 'UNKNOWN')
                }

                # 제품 정보가 있으면 추가
                if 'product_id' in rec:
                    overlay['product_id'] = rec['product_id']
                    overlay['product_name'] = rec.get('product_name', '')

                overlays.append(overlay)

            print(f"  총 {len(overlays)}개 오버레이 생성됨")

        return overlays

    def _save_profile(self):
        """사용자 프로필 저장"""
        try:
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"프로필 저장 오류: {e}")
