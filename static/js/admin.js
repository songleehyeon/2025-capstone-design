/**
 * 관리 페이지 JavaScript
 * 페르소나 설정 및 프로필 업데이트
 */

class AdminPanel {
    constructor() {
        // DOM 요소
        this.personaCards = document.querySelectorAll('.persona-card');
        this.saveBtn = document.getElementById('save-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.statusMessage = document.getElementById('status-message');
        this.profilePreview = document.getElementById('profile-preview');

        // 폼 요소
        this.gender = document.getElementById('gender');
        this.age = document.getElementById('age');
        this.occupation = document.getElementById('occupation');
        this.livingType = document.getElementById('living-type');
        this.allergies = document.getElementById('allergies');
        this.vegan = document.getElementById('vegan');
        this.lowSugar = document.getElementById('low-sugar');
        this.lowCaffeine = document.getElementById('low-caffeine');
        this.attributePreferences = document.getElementById('attribute-preferences');
        this.priceSensitivity = document.getElementById('price-sensitivity');
        this.currentTime = document.getElementById('current-time');
        this.dayType = document.getElementById('day-type');
        this.weather = document.getElementById('weather');

        // 모든 입력 필드 목록
        this.allInputs = [
            this.gender, this.age, this.occupation, this.livingType,
            this.allergies, this.vegan, this.lowSugar, this.lowCaffeine,
            this.attributePreferences, this.priceSensitivity,
            this.currentTime, this.dayType, this.weather
        ];

        // 상태
        this.selectedPersona = 'custom';
        this.personasData = null;

        // API 엔드포인트
        const protocol = window.location.protocol;
        const host = window.location.host;
        this.apiUrl = `${protocol}//${host}/api/update_profile`;
        this.getProfileUrl = `${protocol}//${host}/api/user_profile`;
        this.personasUrl = `${protocol}//${host}/api/personas`;
        this.selectPersonaUrl = `${protocol}//${host}/api/select_persona`;

        this.init();
    }

    async init() {
        // 페르소나 데이터 로드
        await this.loadPersonasData();

        // 페르소나 카드 클릭 이벤트
        this.personaCards.forEach(card => {
            card.addEventListener('click', () => this.selectPersona(card));
        });

        // 버튼 이벤트
        this.saveBtn.addEventListener('click', () => this.saveProfile());
        this.resetBtn.addEventListener('click', () => this.resetProfile());

        // 초기 프로필 로드
        this.loadCurrentProfile();
    }

    async loadPersonasData() {
        try {
            const response = await fetch(this.personasUrl);
            const data = await response.json();
            if (data.success) {
                this.personasData = data.personas;
            } else {
                console.error('페르소나 로드 실패:', data.error);
            }
        } catch (error) {
            console.error('페르소나 데이터 로드 오류:', error);
            this.showStatus('페르소나 데이터 로드 실패', 'error');
        }
    }

    async selectPersona(card) {
        // 모든 카드 비활성화
        this.personaCards.forEach(c => c.classList.remove('active'));

        // 선택한 카드 활성화
        card.classList.add('active');
        const personaId = card.dataset.persona;
        this.selectedPersona = personaId;

        // 커스텀이 아니면 API로 페르소나 선택 요청
        if (personaId !== 'custom') {
            try {
                const response = await fetch(this.selectPersonaUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ persona_id: personaId })
                });

                const result = await response.json();
                if (result.success) {
                    console.log('✓ 페르소나 선택됨:', result.profile.persona_name);
                    this.showStatus(`✓ ${result.profile.persona_name} 페르소나 적용됨`, 'success');

                    // UI에 프로필 반영
                    this.applyPersonaPreset(personaId);
                } else {
                    console.error('페르소나 선택 실패:', result.error);
                    this.showStatus('페르소나 선택 실패', 'error');
                }
            } catch (error) {
                console.error('페르소나 API 호출 오류:', error);
                this.showStatus('페르소나 API 호출 실패', 'error');
            }
        } else {
            // 커스텀인 경우 프리셋 적용
            this.applyPersonaPreset(personaId);
        }

        // 커스텀이 아니면 필드 비활성화
        this.toggleFieldsEditable(personaId === 'custom');
    }

    toggleFieldsEditable(editable) {
        this.allInputs.forEach(input => {
            if (input) {
                input.disabled = !editable;
            }
        });
    }

    applyPersonaPreset(persona) {
        // 커스텀인 경우 현재 값 유지
        if (persona === 'custom') {
            return;
        }

        // personas.json에서 데이터 가져오기
        if (!this.personasData || !this.personasData[persona]) {
            console.error('페르소나 데이터 없음:', persona);
            return;
        }

        const personaData = this.personasData[persona];

        // 기본 정보
        this.gender.value = personaData.demographics.gender || 'female';
        this.age.value = personaData.demographics.age || '';
        this.occupation.value = (personaData.demographics.occupation || []).join(', ');
        this.livingType.value = (personaData.living.type || []).join(', ');

        // 식이 선호도
        this.allergies.value = (personaData.dietary.allergies || []).join(', ');
        this.vegan.checked = personaData.dietary.vegan || false;
        this.lowSugar.checked = personaData.dietary.low_sugar_preference || false;
        this.lowCaffeine.checked = personaData.dietary.low_caffeine_preference || false;

        // 속성 선호도
        this.attributePreferences.value = (personaData.preferences.attribute_preferences || []).join(', ');
        this.priceSensitivity.value = personaData.preferences.price_sensitivity || 'medium';

        // 상황 정보
        this.currentTime.value = personaData.context.current_time || 'afternoon';
        this.dayType.value = personaData.context.day_type || 'weekday';
        this.weather.value = personaData.context.weather || 'normal';
    }

    async loadCurrentProfile() {
        try {
            const response = await fetch(this.getProfileUrl);
            const data = await response.json();

            if (data.success) {
                const profile = data.profile;

                // 폼에 데이터 채우기
                this.gender.value = profile.gender || 'female';
                this.age.value = profile.age || '';
                this.occupation.value = (profile.occupation || []).join(', ');
                this.livingType.value = (profile.living_type || []).join(', ');
                this.allergies.value = (profile.allergies || []).join(', ');
                this.vegan.checked = profile.vegan || false;
                this.lowSugar.checked = profile.low_sugar_preference || false;
                this.lowCaffeine.checked = profile.low_caffeine_preference || false;
                this.attributePreferences.value = (profile.attribute_preferences || []).join(', ');
                this.priceSensitivity.value = profile.price_sensitivity || 'medium';

                // 상황 정보
                const context = profile.context || {};
                this.currentTime.value = context.current_time || 'afternoon';
                this.dayType.value = context.day_type || 'weekday';
                this.weather.value = context.weather || 'normal';

                // 미리보기 업데이트
                this.updatePreview(profile);
            }
        } catch (error) {
            console.error('프로필 로드 오류:', error);
            this.showStatus('프로필 로드 실패', 'error');
        }
    }

    async saveProfile() {
        try {
            // age_group 계산
            const age = parseInt(this.age.value) || 0;
            let ageGroup = '20s';
            if (age >= 60) ageGroup = '60s+';
            else if (age >= 50) ageGroup = '50s';
            else if (age >= 40) ageGroup = '40s';
            else if (age >= 30) ageGroup = '30s';
            else if (age >= 20) ageGroup = '20s';

            // 페르소나 이름 가져오기
            let personaName = '커스텀';
            if (this.selectedPersona !== 'custom' && this.personasData && this.personasData[this.selectedPersona]) {
                personaName = this.personasData[this.selectedPersona].displayName || this.personasData[this.selectedPersona].name;
            }

            // 프로필 객체 생성
            const profile = {
                user_id: 'user01',
                persona_type: this.selectedPersona,
                persona_name: personaName,
                gender: this.gender.value,
                age: age,
                age_group: ageGroup,
                occupation: this.occupation.value.split(',').map(s => s.trim()).filter(s => s),
                living_type: this.livingType.value.split(',').map(s => s.trim()).filter(s => s),
                allergies: this.allergies.value.split(',').map(s => s.trim()).filter(s => s),
                vegan: this.vegan.checked,
                low_sugar_preference: this.lowSugar.checked,
                low_caffeine_preference: this.lowCaffeine.checked,
                price_sensitivity: this.priceSensitivity.value,
                attribute_preferences: this.attributePreferences.value.split(',').map(s => s.trim()).filter(s => s),
                context: {
                    current_time: this.currentTime.value,
                    day_type: this.dayType.value,
                    weather: this.weather.value
                },
                personalization_level: 'high'
            };

            // API 요청
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(profile)
            });

            const data = await response.json();

            if (data.success) {
                this.showStatus('✅ 저장 완료! 스마트폰에 실시간 반영되었습니다.', 'success');
                this.updatePreview(profile);
            } else {
                throw new Error(data.error || '저장 실패');
            }

        } catch (error) {
            console.error('저장 오류:', error);
            this.showStatus('❌ 저장 실패: ' + error.message, 'error');
        }
    }

    resetProfile() {
        if (confirm('정말로 초기화하시겠습니까?')) {
            this.gender.value = 'female';
            this.age.value = '';
            this.occupation.value = '';
            this.livingType.value = '';
            this.allergies.value = '';
            this.vegan.checked = false;
            this.lowSugar.checked = false;
            this.lowCaffeine.checked = false;
            this.attributePreferences.value = '';
            this.priceSensitivity.value = 'medium';
            this.currentTime.value = 'afternoon';
            this.dayType.value = 'weekday';
            this.weather.value = 'normal';

            // 커스텀 페르소나 선택
            this.personaCards.forEach(card => card.classList.remove('active'));
            document.querySelector('[data-persona="custom"]').classList.add('active');
            this.selectedPersona = 'custom';
            this.toggleFieldsEditable(true);

            this.showStatus('초기화되었습니다.', 'success');
        }
    }

    showStatus(message, type) {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;

        // 3초 후 숨김
        setTimeout(() => {
            this.statusMessage.className = 'status-message';
        }, 3000);
    }

    updatePreview(profile) {
        this.profilePreview.textContent = JSON.stringify(profile, null, 2);
    }
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    const admin = new AdminPanel();
    console.log('관리 페이지 초기화 완료');
});
