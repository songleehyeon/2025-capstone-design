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
        this.coffeeTypes = document.getElementById('coffee-types');
        this.allergies = document.getElementById('allergies');
        this.interests = document.getElementById('interests');
        this.vegetarian = document.getElementById('vegetarian');
        this.vegan = document.getElementById('vegan');
        this.priceSensitivity = document.getElementById('price-sensitivity');

        // 상태
        this.selectedPersona = 'custom';

        // API 엔드포인트
        const protocol = window.location.protocol;
        const host = window.location.host;
        this.apiUrl = `${protocol}//${host}/api/update_profile`;
        this.getProfileUrl = `${protocol}//${host}/api/user_profile`;

        this.init();
    }

    init() {
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

    selectPersona(card) {
        // 모든 카드 비활성화
        this.personaCards.forEach(c => c.classList.remove('active'));

        // 선택한 카드 활성화
        card.classList.add('active');
        this.selectedPersona = card.dataset.persona;

        // 프리셋 적용
        this.applyPersonaPreset(this.selectedPersona);
    }

    applyPersonaPreset(persona) {
        const presets = {
            'young-female': {
                coffee: '라떼, 바닐라라떼, 카푸치노',
                allergies: '',
                interests: '패션, 뷰티, 카페, SNS',
                vegetarian: false,
                vegan: false,
                price: 'medium'
            },
            'young-male': {
                coffee: '아메리카노, 콜드브루',
                allergies: '',
                interests: '게임, IT, 전자제품, 운동',
                vegetarian: false,
                vegan: false,
                price: 'medium'
            },
            'middle-female': {
                coffee: '라떼, 디카페인',
                allergies: '',
                interests: '육아, 가전제품, 건강, 홈인테리어',
                vegetarian: false,
                vegan: false,
                price: 'high'
            },
            'middle-male': {
                coffee: '아메리카노, 에스프레소',
                allergies: '',
                interests: '자동차, 투자, 골프, IT',
                vegetarian: false,
                vegan: false,
                price: 'low'
            },
            'senior': {
                coffee: '아메리카노, 디카페인',
                allergies: '',
                interests: '건강, 여행, 골프, 등산',
                vegetarian: false,
                vegan: false,
                price: 'low'
            },
            'custom': {
                // 현재 입력값 유지
                coffee: this.coffeeTypes.value,
                allergies: this.allergies.value,
                interests: this.interests.value,
                vegetarian: this.vegetarian.checked,
                vegan: this.vegan.checked,
                price: this.priceSensitivity.value
            }
        };

        const preset = presets[persona];
        if (preset) {
            this.coffeeTypes.value = preset.coffee;
            this.allergies.value = preset.allergies;
            this.interests.value = preset.interests;
            this.vegetarian.checked = preset.vegetarian;
            this.vegan.checked = preset.vegan;
            this.priceSensitivity.value = preset.price;
        }
    }

    async loadCurrentProfile() {
        try {
            const response = await fetch(this.getProfileUrl);
            const data = await response.json();

            if (data.success) {
                const profile = data.profile;

                // 폼에 데이터 채우기
                const prefs = profile.preferences || {};
                this.coffeeTypes.value = (prefs.coffee_type || []).join(', ');
                this.allergies.value = (prefs.dietary?.allergies || []).join(', ');
                this.interests.value = (prefs.interests || []).join(', ');
                this.vegetarian.checked = prefs.dietary?.vegetarian || false;
                this.vegan.checked = prefs.dietary?.vegan || false;
                this.priceSensitivity.value = prefs.price_sensitivity || 'medium';

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
            // 프로필 객체 생성
            const profile = {
                user_id: 'user001',
                preferences: {
                    coffee_type: this.coffeeTypes.value.split(',').map(s => s.trim()).filter(s => s),
                    dietary: {
                        allergies: this.allergies.value.split(',').map(s => s.trim()).filter(s => s),
                        vegetarian: this.vegetarian.checked,
                        vegan: this.vegan.checked
                    },
                    price_sensitivity: this.priceSensitivity.value,
                    interests: this.interests.value.split(',').map(s => s.trim()).filter(s => s)
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
            this.coffeeTypes.value = '';
            this.allergies.value = '';
            this.interests.value = '';
            this.vegetarian.checked = false;
            this.vegan.checked = false;
            this.priceSensitivity.value = 'medium';

            // 커스텀 페르소나 선택
            this.personaCards.forEach(card => card.classList.remove('active'));
            document.querySelector('[data-persona="custom"]').classList.add('active');
            this.selectedPersona = 'custom';

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
