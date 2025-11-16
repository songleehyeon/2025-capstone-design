"""
개인화 로직 모듈
사용자 프로필과 OCR 텍스트를 기반으로 개인화된 콘텐츠를 생성합니다.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path


class PersonalizationEngine:
    """개인화 콘텐츠 생성 엔진"""

    def __init__(self, profile_path: str = "config/user_profile.json"):
        """
        초기화

        Args:
            profile_path: 사용자 프로필 JSON 파일 경로
        """
        # 상대 경로를 프로젝트 루트 기준으로 변환
        if not Path(profile_path).is_absolute():
            # src 폴더에서 실행해도 작동하도록 프로젝트 루트 찾기
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent  # src의 부모 = capstone
            profile_path = str(project_root / profile_path)

        self.profile_path = profile_path
        self.user_profile = self._load_profile()

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
            "preferences": {
                "coffee_type": [],
                "dietary": {"allergies": [], "vegetarian": False, "vegan": False},
                "price_sensitivity": "medium",
                "favorite_brands": [],
                "interests": []
            },
            "history": {
                "recent_purchases": [],
                "favorite_items": []
            },
            "personalization_level": "medium"
        }

    def analyze_text_content(self, ocr_results: List[Dict]) -> Dict:
        """
        OCR 텍스트 내용 분석

        Args:
            ocr_results: OCR 결과 리스트

        Returns:
            분석 결과 (카테고리, 키워드 등)
        """
        all_text = ' '.join([r['text'] for r in ocr_results])

        analysis = {
            'categories': [],
            'items': [],
            'prices': [],
            'keywords': []
        }

        # 카페/음료 관련 키워드
        cafe_keywords = ['아메리카노', '라떼', '카페', '커피', '에스프레소',
                         '카푸치노', '마키아또', '디카페인', '음료']

        # 베이커리 관련 키워드
        bakery_keywords = ['빵', '케이크', '크루아상', '머핀', '쿠키',
                          '디저트', '베이글']

        # 가격 패턴
        import re
        price_pattern = r'\d{1,3}(?:,\d{3})*원?'
        prices = re.findall(price_pattern, all_text)

        # 카테고리 감지
        if any(keyword in all_text for keyword in cafe_keywords):
            analysis['categories'].append('cafe')

        if any(keyword in all_text for keyword in bakery_keywords):
            analysis['categories'].append('bakery')

        # 개별 항목 추출
        for result in ocr_results:
            text = result['text']

            # 메뉴 항목 감지
            if any(keyword in text for keyword in cafe_keywords + bakery_keywords):
                analysis['items'].append({
                    'name': text,
                    'bbox': result['bbox'],
                    'center': result['center'],
                    'confidence': result['confidence']
                })

        # 가격 정보
        analysis['prices'] = prices

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
        preferences = self.user_profile.get('preferences', {})
        history = self.user_profile.get('history', {})

        # 선호 음료 강조
        coffee_prefs = preferences.get('coffee_type', [])
        for item in content_analysis.get('items', []):
            item_name = item['name']

            # 선호 메뉴 하이라이트
            if any(pref in item_name for pref in coffee_prefs):
                overlays.append({
                    'type': 'highlight',
                    'position': item['center'],
                    'bbox': item['bbox'],
                    'content': '💚 추천',
                    'color': (0, 255, 0),
                    'reason': '선호 메뉴'
                })

            # 최근 구매 항목
            recent = history.get('recent_purchases', [])
            if any(rec in item_name for rec in recent):
                overlays.append({
                    'type': 'badge',
                    'position': item['center'],
                    'bbox': item['bbox'],
                    'content': '🔁 재구매',
                    'color': (255, 165, 0),
                    'reason': '최근 구매 이력'
                })

            # 알레르기 경고
            allergies = preferences.get('dietary', {}).get('allergies', [])
            for allergen in allergies:
                if allergen in item_name:
                    overlays.append({
                        'type': 'warning',
                        'position': item['center'],
                        'bbox': item['bbox'],
                        'content': f'⚠️ {allergen} 주의',
                        'color': (0, 0, 255),
                        'reason': '알레르기 성분 포함'
                    })

        # 할인/프로모션 정보 추가 (시뮬레이션) - 제거됨
        # if len(overlays) < 3:  # 오버레이가 적을 때 프로모션 추가
        #     overlays.append({
        #         'type': 'promotion',
        #         'position': [100, 50],
        #         'content': '🎁 신규 고객 10% 할인',
        #         'color': (255, 0, 255),
        #         'reason': '프로모션'
        #     })

        return overlays

    def get_recommendation_score(self, item_name: str) -> float:
        """
        특정 항목의 추천 점수 계산

        Args:
            item_name: 항목 이름

        Returns:
            추천 점수 (0.0 ~ 1.0)
        """
        score = 0.5  # 기본 점수

        preferences = self.user_profile.get('preferences', {})
        history = self.user_profile.get('history', {})

        # 선호 타입 매칭
        coffee_prefs = preferences.get('coffee_type', [])
        if any(pref in item_name for pref in coffee_prefs):
            score += 0.3

        # 최근 구매 이력
        recent = history.get('recent_purchases', [])
        if any(rec in item_name for rec in recent):
            score += 0.2

        # 즐겨찾기
        favorites = history.get('favorite_items', [])
        if any(fav in item_name for fav in favorites):
            score += 0.4

        # 알레르기 체크 (점수 감소)
        allergies = preferences.get('dietary', {}).get('allergies', [])
        if any(allergen in item_name for allergen in allergies):
            score -= 0.5

        return max(0.0, min(1.0, score))  # 0~1 범위로 클리핑

    def filter_by_dietary(self, items: List[str]) -> List[str]:
        """
        식이 제한에 따른 항목 필터링

        Args:
            items: 항목 이름 리스트

        Returns:
            필터링된 항목 리스트
        """
        dietary = self.user_profile.get('preferences', {}).get('dietary', {})
        allergies = dietary.get('allergies', [])
        is_vegetarian = dietary.get('vegetarian', False)
        is_vegan = dietary.get('vegan', False)

        filtered = []

        for item in items:
            # 알레르기 성분 체크
            if any(allergen in item for allergen in allergies):
                continue

            # 채식주의 체크 (간단한 키워드 기반)
            if is_vegan:
                non_vegan_keywords = ['우유', '치즈', '버터', '크림', '요거트']
                if any(keyword in item for keyword in non_vegan_keywords):
                    continue

            filtered.append(item)

        return filtered

    def update_history(self, item_name: str, action: str = 'view'):
        """
        사용자 행동 이력 업데이트

        Args:
            item_name: 항목 이름
            action: 행동 타입 ('view', 'purchase', 'favorite')
        """
        history = self.user_profile.get('history', {})

        if action == 'purchase':
            recent = history.get('recent_purchases', [])
            if item_name not in recent:
                recent.insert(0, item_name)
                history['recent_purchases'] = recent[:10]  # 최근 10개만 유지

        elif action == 'favorite':
            favorites = history.get('favorite_items', [])
            if item_name not in favorites:
                favorites.append(item_name)
                history['favorite_items'] = favorites

        # 프로필 저장 (실제 애플리케이션에서는 서버에 저장)
        self._save_profile()

    def _save_profile(self):
        """사용자 프로필 저장"""
        try:
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"프로필 저장 오류: {e}")

    def get_profile_summary(self) -> str:
        """
        사용자 프로필 요약 정보

        Returns:
            프로필 요약 문자열
        """
        prefs = self.user_profile.get('preferences', {})
        coffee = ', '.join(prefs.get('coffee_type', []))
        allergies = ', '.join(prefs.get('dietary', {}).get('allergies', []))

        summary = f"""
사용자 프로필 요약:
- ID: {self.user_profile.get('user_id', 'Unknown')}
- 선호 음료: {coffee if coffee else '없음'}
- 알레르기: {allergies if allergies else '없음'}
- 개인화 수준: {self.user_profile.get('personalization_level', 'medium')}
        """

        return summary.strip()
