"""
규칙 기반 제품 추천 시스템
사용자 페르소나와 상황 정보를 기반으로 제품 추천 및 정보 제공
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import random


class RecommendationEngine:
    """규칙 기반 추천 엔진"""
    def __init__(self, products_path: str = "config/starbucks_products.json"):
        if not Path(products_path).is_absolute():
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            products_path = str(project_root / products_path)

        self.products_path = products_path
        self.products_data = self._load_products()

        # random 
        self.crowd_levels = ["low", "medium", "high", "very_high"]
        self.wait_times = {"low": 3, "medium": 5, "high": 8, "very_high": 12}

        print("✓ RecommendationEngine 초기화 완료")

    def _load_products(self) -> Dict:
        """제품 정보 로드"""
        try:
            with open(self.products_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"경고: 제품 파일을 찾을 수 없습니다: {self.products_path}")
            return {"products": {}}
        except json.JSONDecodeError:
            print(f"경고: 제품 파일 형식 오류: {self.products_path}")
            return {"products": {}}

    def get_recommendations(self, user_profile: Dict) -> List[Dict]:
        """
        사용자 프로필에 기반한 추천 생성

        Args:
            user_profile: 사용자 프로필 정보

        Returns:
            추천 오버레이 리스트 (우선순위 순)
        """
        recommendations = []

        # 1. WARN 규칙 (최우선)
        recommendations.extend(self._apply_warn_rules(user_profile))

        # 2. GENERAL_RECOMM 규칙
        recommendations.extend(self._apply_general_recomm_rules(user_profile))

        # 3. EVENT_INFO 규칙
        recommendations.extend(self._apply_event_info_rules(user_profile))

        # 4. WEATHER_RECOMM 규칙
        recommendations.extend(self._apply_weather_recomm_rules(user_profile))

        # 5. CROWD_LEVEL 규칙
        recommendations.extend(self._apply_crowd_level_rules(user_profile))

        # 우선순위 정렬 (priority 높은 순, 같으면 score 높은 순)
        recommendations.sort(key=lambda x: (x.get('priority', 0), x.get('score', 0)), reverse=True)

        return recommendations

    def _apply_warn_rules(self, user_profile: Dict) -> List[Dict]:
        """경고 규칙 적용"""
        warnings = []
        products = self.products_data.get('products', {})
        allergies = user_profile.get('allergies', [])
        low_caffeine_pref = user_profile.get('low_caffeine_preference', False)

        for product_id, product in products.items():
            # 글루텐 알레르기 경고
            if 'gluten' in allergies and 'gluten' in product.get('allergies', []):
                warnings.append({
                    'rule_id': 'WARN',
                    'priority': 100,  # prior
                    'score': 0,
                    'product_id': product_id,
                    'product_name': product['name'],
                    'message': f"⚠️ 주의! 글루텐이 들어간 제품이에요",
                    'type': 'warning',
                    'color': (255, 200, 200)  # 부드러운 연한 빨강 (글래스모피즘)
                })

            # 고카페인 경고 (디카페인 옵션 선호자)
            if low_caffeine_pref and product.get('caffeine') == 'high' and 'decaf' in product.get('options', []):
                warnings.append({
                    'rule_id': 'WARN',
                    'priority': 100,  # prior
                    'score': 0,
                    'product_id': product_id,
                    'product_name': product['name'],
                    'message': f"⚠️ 주의! 고카페인이 함유된 제품이에요\n💡 디카페인 옵션으로 변경 가능",
                    'type': 'warning',
                    'color': (255, 220, 200)  # 부드러운 연한 주황 (글래스모피즘)
                })

        return warnings

    def _apply_general_recomm_rules(self, user_profile: Dict) -> List[Dict]:
        """일반 추천 규칙 적용"""
        recommendations = []
        products = self.products_data.get('products', {})
        gender = user_profile.get('gender')
        age_group = user_profile.get('age_group')

        for product_id, product in products.items():
            target_demo = product.get('target_demographics', {})
            target_gender = target_demo.get('gender', [])
            target_age_groups = target_demo.get('age_group', [])

            # 성별 + 연령대 매칭
            if gender in target_gender and age_group in target_age_groups:
                # 메시지 생성
                if age_group in ['20s', '30s'] and gender == 'female':
                    message = f"💁‍♀️ 2-30대 여성이 많이 주문한 메뉴예요"
                elif age_group == '10s' and gender == 'female':
                    message = f"💁‍♀️ 10대 여성이 많이 주문한 메뉴예요"
                elif age_group in ['50s', '60s+']:
                    message = f"👥 50대 이상이 많이 주문한 메뉴예요"
                else:
                    continue

                recommendations.append({
                    'rule_id': 'GENERAL_RECOMM',
                    'priority': 20,
                    'score': 0,
                    'product_id': product_id,
                    'product_name': product['name'],
                    'message': message,
                    'type': 'recommendation',
                    'color': (200, 240, 200)  # 부드러운 연한 민트 (글래스모피즘)
                })

        return recommendations

    def _apply_event_info_rules(self, user_profile: Dict) -> List[Dict]:
        """이벤트 정보 규칙 적용"""
        events = []
        price_sensitivity = user_profile.get('price_sensitivity', 'medium')
        occupation = user_profile.get('occupation', [])

        # 프리퀀시 정보
        if price_sensitivity == 'high':
            priority = 30
        elif price_sensitivity == 'medium':
            priority = 20
        else:
            priority = 0

        if priority > 0:
            events.append({
                'rule_id': 'EVENT_INFO',
                'priority': priority,
                'score': 0,
                'message': f"🎁 시즌 음료를 구입하면 프리퀀시가 3개 남아요!",
                'type': 'event',
                'color': (255, 245, 220)  # 부드러운 연한 노랑 (글래스모피즘)
            })

        # 학생 할인
        if 'student' in occupation:
            if price_sensitivity == 'high':
                priority = 30
            elif price_sensitivity == 'medium':
                priority = 20
            else:
                priority = 0

            if priority > 0:
                events.append({
                    'rule_id': 'EVENT_INFO',
                    'priority': priority,
                    'score': 0,
                    'message': f"🎓 학생 할인쿠폰이 적용가능한 상품이에요",
                    'type': 'event',
                    'color': (220, 230, 255)  # 부드러운 연한 파랑 (글래스모피즘)
                })

        return events

    def _apply_weather_recomm_rules(self, user_profile: Dict) -> List[Dict]:
        """날씨 기반 추천 규칙 적용"""
        recommendations = []
        context = user_profile.get('context', {})
        weather = context.get('weather', 'normal')
        current_time = context.get('current_time', 'afternoon')
        attr_prefs = user_profile.get('attribute_preferences', [])
        products = self.products_data.get('products', {})

        # 추운 날씨 + 라떼 추천
        if weather == 'cold':
            has_latte = 'latte' in attr_prefs
            has_hot = 'hot' in attr_prefs

            if has_latte and has_hot:
                priority = 30
            elif has_latte or has_hot:
                priority = 20
            else:
                priority = 10

            recommendations.append({
                'rule_id': 'WEATHER_RECOMM',
                'priority': priority,
                'score': 0,
                'message': f"❄️ 쌀쌀한 날씨, 따뜻한 라떼는 어떠세요?",
                'type': 'recommendation',
                'color': (210, 230, 255)  # 부드러운 하늘색 (글래스모피즘)
            })

        # 오후 시간 + 달콤한 메뉴 추천 (말차 초콜릿 라떼)
        if current_time == 'afternoon':
            has_sweet = 'sweet' in attr_prefs
            has_matcha = 'matcha' in attr_prefs

            if has_sweet and has_matcha:
                priority = 30
            elif has_sweet:
                priority = 20
            else:
                priority = 10

            # P2: 말차 초콜릿 라떼
            if 'P2' in products:
                recommendations.append({
                    'rule_id': 'WEATHER_RECOMM',
                    'priority': priority,
                    'score': 0,
                    'product_id': 'P2',
                    'product_name': products['P2']['name'],
                    'message': f"🍫 오후 시간, 달콤한 메뉴로 당충전 어떠세요?",
                    'type': 'recommendation',
                    'color': (240, 220, 255)  # 부드러운 연한 보라 (글래스모피즘)
                })

        return recommendations

    def _apply_crowd_level_rules(self, user_profile: Dict) -> List[Dict]:
        """혼잡도 규칙 적용"""
        crowd_info = []

        # 혼잡도 시뮬레이션 (실제로는 store.crowd_level 사용)
        store_info = self.products_data.get('store', {})
        crowd_level = store_info.get('crowd_level', 'medium')

        # 혼잡도에 따른 우선순위와 메시지
        crowd_messages = {
            'very_high': {
                'priority': 100,  # prior
                'message': f"🚨 혼잡도: 매우 혼잡! 대기 시간: {self.wait_times['very_high']}분",
                'color': (255, 210, 210)  # 부드러운 연한 빨강 (글래스모피즘)
            },
            'high': {
                'priority': 30,
                'message': f"⚠️ 혼잡도: 혼잡 대기 시간: {self.wait_times['high']}분",
                'color': (255, 235, 210)  # 부드러운 연한 주황 (글래스모피즘)
            },
            'medium': {
                'priority': 20,
                'message': f"ℹ️ 혼잡도: 보통 대기 시간: {self.wait_times['medium']}분",
                'color': (230, 245, 230)  # 부드러운 연한 민트 (글래스모피즘)
            },
            'low': {
                'priority': 10,
                'message': f"✅ 혼잡도: 여유 대기 시간: {self.wait_times['low']}분",
                'color': (220, 250, 220)  # 부드러운 연한 초록 (글래스모피즘)
            }
        }

        if crowd_level in crowd_messages:
            info = crowd_messages[crowd_level]
            crowd_info.append({
                'rule_id': 'CROWD_LEVEL',
                'priority': info['priority'],
                'score': 0,
                'message': info['message'],
                'type': 'info',
                'color': info['color']
            })

        return crowd_info

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """제품 ID로 제품 정보 가져오기"""
        return self.products_data.get('products', {}).get(product_id)

    def get_all_products(self) -> Dict:
        """모든 제품 정보 가져오기"""
        return self.products_data.get('products', {})
