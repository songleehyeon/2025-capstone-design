"""
통합 규칙 기반 추천 엔진
설정 파일 기반으로 다양한 광고에 대한 추천을 생성합니다.
"""

import json
from typing import List, Dict, Optional, Any
from pathlib import Path


class UniversalRecommendationEngine:
    """설정 기반 통합 추천 엔진"""

    def __init__(self, ads_config_dir: str = "config/ads"):
        """
        초기화

        Args:
            ads_config_dir: 광고 설정 파일 디렉토리 경로
        """
        if not Path(ads_config_dir).is_absolute():
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            ads_config_dir = str(project_root / ads_config_dir)

        self.ads_config_dir = Path(ads_config_dir)
        self.ad_configs = self._load_all_ads()

        print(f"✓ UniversalRecommendationEngine 초기화 완료")
        print(f"  로드된 광고 개수: {len(self.ad_configs)}")
        print(f"  광고 목록: {', '.join(self.ad_configs.keys())}")

    def _load_all_ads(self) -> Dict[str, Dict]:
        """모든 광고 설정 파일 로드"""
        ad_configs = {}

        if not self.ads_config_dir.exists():
            print(f"경고: 광고 설정 디렉토리를 찾을 수 없습니다: {self.ads_config_dir}")
            return ad_configs

        for config_file in self.ads_config_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    brand = config.get('brand')
                    if brand:
                        ad_configs[brand] = config
                        print(f"  ✓ {brand} 광고 설정 로드됨")
            except Exception as e:
                print(f"  ✗ {config_file.name} 로드 실패: {e}")

        return ad_configs

    def detect_brand(self, ocr_text: str) -> Optional[str]:
        """
        OCR 텍스트에서 브랜드 감지

        Args:
            ocr_text: OCR로 추출된 텍스트

        Returns:
            감지된 브랜드명 (없으면 None)
        """
        ocr_text_lower = ocr_text.lower()

        for brand, config in self.ad_configs.items():
            keywords = config.get('detection_keywords', [])
            if any(keyword.lower() in ocr_text_lower for keyword in keywords):
                print(f"✓ {brand} 브랜드 감지됨! (키워드 매칭)")
                return brand

        return None

    def get_recommendations(self, brand: str, user_profile: Dict) -> List[Dict]:
        """
        브랜드별 추천 생성

        Args:
            brand: 브랜드명
            user_profile: 사용자 프로필

        Returns:
            추천 오버레이 리스트 (우선순위 순 정렬)
        """
        if brand not in self.ad_configs:
            print(f"경고: {brand} 브랜드 설정을 찾을 수 없습니다.")
            return []

        config = self.ad_configs[brand]
        recommendations = []

        # 각 룰 평가
        for rule in config.get('rules', []):
            # 조건 평가
            if self._evaluate_conditions(rule.get('conditions', {}), user_profile, config):
                rec = {
                    'rule_id': rule.get('rule_id', 'UNKNOWN'),
                    'priority': rule.get('priority', 0),
                    'type': rule.get('type', 'info'),
                    'color': tuple(rule.get('color', [100, 200, 100]))
                }

                # 메시지 생성
                if 'message_template' in rule:
                    # 템플릿 기반 메시지는 나중에 처리
                    rec['message'] = rule['message_template']
                elif 'message_templates' in rule:
                    # 조건에 따른 메시지 선택
                    rec['message'] = self._select_message_template(
                        rule['message_templates'],
                        user_profile
                    )
                else:
                    rec['message'] = rule.get('message', '')

                # 제품 정보 추가 (있는 경우)
                if 'product_id' in rule:
                    product_id = rule['product_id']
                    rec['product_id'] = product_id
                    products = config.get('products', {})
                    if product_id in products:
                        rec['product_name'] = products[product_id].get('name', '')

                recommendations.append(rec)

        # 우선순위 정렬
        recommendations.sort(key=lambda x: x['priority'], reverse=True)

        return recommendations

    def _evaluate_conditions(self, conditions: Dict, user_profile: Dict, ad_config: Dict) -> bool:
        """
        조건 평가

        Args:
            conditions: 조건 딕셔너리
            user_profile: 사용자 프로필
            ad_config: 광고 설정

        Returns:
            조건 충족 여부
        """
        condition_type = conditions.get('type')

        if condition_type == 'always_true':
            return True

        elif condition_type == 'equals':
            user_field = conditions.get('user_field')
            value = conditions.get('value')
            return user_profile.get(user_field) == value

        elif condition_type == 'in_array':
            user_field = conditions.get('user_field')
            values = conditions.get('values', [])
            user_value = user_profile.get(user_field)
            return user_value in values

        elif condition_type == 'array_contains':
            user_field = conditions.get('user_field')
            value = conditions.get('value')
            user_array = user_profile.get(user_field, [])
            return value in user_array

        elif condition_type == 'context_equals':
            field = conditions.get('field')
            value = conditions.get('value')
            context = user_profile.get('context', {})
            return context.get(field) == value

        elif condition_type == 'store_field_equals':
            field = conditions.get('field')
            value = conditions.get('value')
            store = ad_config.get('store', {})
            return store.get(field) == value

        elif condition_type == 'and':
            rules = conditions.get('rules', [])
            return all(self._evaluate_conditions(rule, user_profile, ad_config) for rule in rules)

        elif condition_type == 'or':
            rules = conditions.get('rules', [])
            return any(self._evaluate_conditions(rule, user_profile, ad_config) for rule in rules)

        elif condition_type == 'product_field_equals':
            # 제품 속성 확인 (모든 제품 검사)
            field = conditions.get('field')
            value = conditions.get('value')
            products = ad_config.get('products', {})
            return any(product.get(field) == value for product in products.values())

        elif condition_type == 'product_has_option':
            # 제품에 특정 옵션이 있는지 확인
            option = conditions.get('option')
            products = ad_config.get('products', {})
            return any(option in product.get('options', []) for product in products.values())

        elif condition_type == 'allergy_match':
            # 알레르기 매칭 (사용자 알레르기와 제품 알레르기 교집합 확인)
            user_allergies = user_profile.get('allergies', [])
            products = ad_config.get('products', {})

            for product in products.values():
                product_allergies = product.get('allergies', [])
                if any(allergen in product_allergies for allergen in user_allergies):
                    return True
            return False

        elif condition_type == 'product_demographic_match':
            # 제품의 타겟 인구통계와 사용자 매칭
            match_gender = conditions.get('match_gender', False)
            match_age_group = conditions.get('match_age_group', False)

            user_gender = user_profile.get('gender')
            user_age_group = user_profile.get('age_group')

            products = ad_config.get('products', {})

            for product in products.values():
                target_demo = product.get('target_demographics', {})

                gender_match = True
                age_match = True

                if match_gender:
                    target_genders = target_demo.get('gender', [])
                    gender_match = user_gender in target_genders

                if match_age_group:
                    target_ages = target_demo.get('age_group', [])
                    age_match = user_age_group in target_ages

                if gender_match and age_match:
                    return True

            return False

        else:
            print(f"경고: 알 수 없는 조건 타입: {condition_type}")
            return False

    def _select_message_template(self, templates: Dict, user_profile: Dict) -> str:
        """
        조건에 맞는 메시지 템플릿 선택

        Args:
            templates: 메시지 템플릿 딕셔너리
            user_profile: 사용자 프로필

        Returns:
            선택된 메시지
        """
        gender = user_profile.get('gender')
        age_group = user_profile.get('age_group')

        # 성별 + 연령대 조합으로 키 생성
        if gender == 'female' and age_group in ['20s', '30s']:
            key = 'female_20s_30s'
        elif gender == 'female' and age_group == '10s':
            key = 'female_10s'
        elif age_group in ['50s', '60s+']:
            key = '50s_60s'
        else:
            key = 'default'

        return templates.get(key, templates.get('default', ''))

    def get_ad_config(self, brand: str) -> Optional[Dict]:
        """
        특정 브랜드의 광고 설정 가져오기

        Args:
            brand: 브랜드명

        Returns:
            광고 설정 딕셔너리
        """
        return self.ad_configs.get(brand)

    def get_all_brands(self) -> List[str]:
        """
        모든 브랜드 목록 가져오기

        Returns:
            브랜드명 리스트
        """
        return list(self.ad_configs.keys())
