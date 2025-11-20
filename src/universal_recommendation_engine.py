import json
import random
from typing import List, Dict, Optional, Any
from pathlib import Path


class UniversalRecommendationEngine:
    """설정 기반 통합 추천 엔진 (Priority Sort & Deterministic Random Tie-break 적용)"""

    def __init__(self, ads_config_dir: str = "config/ads"):
        if not Path(ads_config_dir).is_absolute():
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            ads_config_dir = str(project_root / ads_config_dir)

        self.ads_config_dir = Path(ads_config_dir)
        self.ad_configs = self._load_all_ads()
        print(f"✓ UniversalRecommendationEngine 초기화 완료 (Loaded: {len(self.ad_configs)} brands)")

    def _load_all_ads(self) -> Dict[str, Dict]:
        ad_configs = {}
        if not self.ads_config_dir.exists():
            return ad_configs

        for config_file in self.ads_config_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    brand = config.get('brand')
                    if brand:
                        ad_configs[brand] = config
            except Exception as e:
                print(f"  ✗ {config_file.name} 로드 실패: {e}")
        return ad_configs

    def detect_brand(self, ocr_text: str) -> Optional[str]:
        ocr_text_lower = ocr_text.lower()
        for brand, config in self.ad_configs.items():
            keywords = config.get('detection_keywords', [])
            if any(keyword.lower() in ocr_text_lower for keyword in keywords):
                return brand
        return None

    def get_recommendations(self, brand: str, user_profile: Dict) -> List[Dict]:
        """
        브랜드별 추천 생성 (Priority Sort + Seeded Random Tie-break + Max 4)
        """
        if brand not in self.ad_configs:
            return []

        config = self.ad_configs[brand]
        valid_recommendations = []

        # 1. 모든 룰 평가하여 조건 만족하는 것 수집
        for rule in config.get('rules', []):
            if self._evaluate_conditions(rule.get('conditions', {}), user_profile, config):
                rec = {
                    'rule_id': rule.get('rule_id', 'UNKNOWN'),
                    'priority': rule.get('priority', 0),
                    'type': rule.get('type', 'info'),
                    'color': tuple(rule.get('color', [100, 200, 100])),
                    'message': rule.get('message', '')
                }
                
                # 템플릿 메시지 처리
                if 'message_template' in rule:
                    rec['message'] = rule['message_template'] 
                elif 'message_templates' in rule:
                    rec['message'] = self._select_message_template(rule['message_templates'], user_profile)

                if 'product_id' in rule:
                    rec['product_id'] = rule['product_id']
                    products = config.get('products', {})
                    if rec['product_id'] in products:
                        rec['product_name'] = products[rec['product_id']].get('name', '')

                valid_recommendations.append(rec)

        # 2. Priority 기준으로 그룹화
        priority_map = {}
        for rec in valid_recommendations:
            p = rec['priority']
            if p not in priority_map:
                priority_map[p] = []
            priority_map[p].append(rec)

        # 3. [핵심 수정] 고정 시드 난수 생성기 사용
        # 사용자 ID와 브랜드명을 조합하여 시드(Seed)를 만듭니다.
        # 이 시드가 같으면 랜덤 셔플 결과도 항상 똑같습니다. -> 깜빡임 해결!
        user_id = user_profile.get('user_id', 'guest')
        seed_value = f"{user_id}_{brand}"
        rng = random.Random(seed_value) 

        # 4. Priority 내림차순으로 순회하며 최대 4개 추출
        final_recommendations = []
        sorted_priorities = sorted(priority_map.keys(), reverse=True)
        MAX_ITEMS = 4

        for p in sorted_priorities:
            if len(final_recommendations) >= MAX_ITEMS:
                break
            
            items = priority_map[p]
            remaining_slots = MAX_ITEMS - len(final_recommendations)
            
            # 동점 항목들을 고정된 패턴으로 섞음 (Seeded Shuffle)
            rng.shuffle(items)

            if len(items) <= remaining_slots:
                final_recommendations.extend(items)
            else:
                # 공간이 부족하면 섞인 순서대로 잘라서 넣음
                final_recommendations.extend(items[:remaining_slots])

        return final_recommendations

    def _evaluate_conditions(self, conditions: Dict, user_profile: Dict, ad_config: Dict) -> bool:
        condition_type = conditions.get('type')

        if condition_type == 'always_true':
            return True
        
        elif condition_type == 'and':
            return all(self._evaluate_conditions(r, user_profile, ad_config) for r in conditions.get('rules', []))
            
        elif condition_type == 'or':
            return any(self._evaluate_conditions(r, user_profile, ad_config) for r in conditions.get('rules', []))

        elif condition_type == 'equals':
            user_field = conditions.get('user_field')
            return user_profile.get(user_field) == conditions.get('value')

        elif condition_type == 'in_array':
            user_field = conditions.get('user_field')
            return user_profile.get(user_field) in conditions.get('values', [])

        elif condition_type == 'array_contains':
            user_field = conditions.get('user_field')
            field_key = 'attribute_preferences' if user_field == 'attributes' else user_field
            user_array = user_profile.get(field_key, [])
            return conditions.get('value') in user_array

        elif condition_type == 'context_equals':
            field = conditions.get('field')
            value = conditions.get('value')
            context = user_profile.get('context', {})
            return context.get(field) == value

        elif condition_type == 'store_field_equals':
            field = conditions.get('field')
            store = ad_config.get('store', {})
            return store.get(field) == conditions.get('value')

        return False

    def _select_message_template(self, templates: Dict, user_profile: Dict) -> str:
        gender = user_profile.get('gender')
        age_group = user_profile.get('age_group')
        
        if gender == 'female' and age_group in ['20s', '30s']:
            return templates.get('female_20s_30s', templates.get('default', ''))
        elif age_group == '10s':
            return templates.get('10s', templates.get('default', ''))
        elif age_group in ['50s', '60s+']:
            return templates.get('50s_senior', templates.get('default', ''))
        
        return templates.get('default', '')