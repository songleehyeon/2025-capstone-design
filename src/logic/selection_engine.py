class AdSelectionEngine:
    def __init__(self, ad_database):
        self.ad_db = ad_database

    def select_ad(self, dominant_group, context_tags):
        """
        우선순위에 따라 광고를 선정합니다.
        1순위: Dominant Group
        2순위: Context Tags (날씨, 시간 등)
        3순위: 기본 광고 ('all')
        
        Args:
            dominant_group (str): e.g., "20s_female"
            context_tags (list): e.g., ["rainy_day", "morning_rush"]

        Returns:
            str: 송출할 광고 파일 경로 (e.g., "assets/ads/olive_young.mp4")
        """
        
        # [TODO] 1순위: Dominant Group 태그와 일치하는 광고 검색
        ad_path = self.find_ad_by_tag(dominant_group)
        if ad_path:
            return ad_path, "Targeted (Crowd)"
        
        # [TODO] 2순위: Context 태그와 일치하는 광고 검색
        for tag in context_tags:
            ad_path = self.find_ad_by_tag(tag)
            if ad_path:
                return ad_path, f"Targeted (Context: {tag})"
                
        # [TODO] 3순위: 'all' 태그가 붙은 기본 광고 검색
        ad_path = self.find_ad_by_tag("all")
        if ad_path:
            return ad_path, "Default (All)"
            
        return None, "No Ad Found"

    def find_ad_by_tag(self, tag):
        for ad_id, ad_info in self.ad_db.items():
            if tag in ad_info["tags"]:
                return ad_info["file_path"]
        return None