class AdminPanel {
    constructor() {
        // 뷰 영역
        this.viewMain = document.getElementById('view-main');
        this.viewCustom = document.getElementById('view-custom');
        
        // 커스텀 폼 단계
        this.step1 = document.getElementById('step-1');
        this.step2 = document.getElementById('step-2');

        // 요소들
        this.personaCards = document.querySelectorAll('.persona-card');
        this.saveCustomBtn = document.getElementById('save-custom-btn');
        this.statusMessage = document.getElementById('status-message');
        this.profilePreview = document.getElementById('profile-preview');

        // API 설정
        const protocol = window.location.protocol;
        const host = window.location.host;
        this.apiUrlSelect = `${protocol}//${host}/api/select_persona`;
        this.apiUrlUpdate = `${protocol}//${host}/api/update_profile`;
        this.apiUrlProfile = `${protocol}//${host}/api/user_profile`;

        // [수정됨] 랜덤 배정될 속성 풀 ("sensitive" 추가됨)
        this.randomAttributesPool = [
            "sensitive", // [이동됨] 민감성 피부는 랜덤
            "wicked_1",
            "latte", "gift_expire_000", "gift_000", 
            "member_1080", "not_member",
            "miss", "kt", "director",
            "meet_백온유", "one_day_left",
            "고양이와 스프", "download_김승규", "로스트아크", "리니지", "검은사막",
            "not_member_gym",
            "hera", "youtuber", "has_black_cushion",
            "wedding_27", "has_history", "meeting", "no_history"
        ];

        this.groupMapping = {
            "story_genre": ["story", "fantasy"],
            "graphic_action": ["graphics", "action"],
            "feminine_chic": ["feminine", "chic"],
            "classic_minimal": ["classic", "minimal"],
            "sporty": ["xexymix", "andar"]
        };

        this.init();
    }

    init() {
        this.personaCards.forEach(card => {
            card.addEventListener('click', () => this.handleCardClick(card));
        });

        this.saveCustomBtn.addEventListener('click', () => this.generateAndSaveCustomProfile());
        this.loadCurrentProfile();
    }

    handleCardClick(card) {
        this.personaCards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');

        const personaId = card.dataset.persona;
        if (personaId === 'custom') {
            this.showCustomView();
        } else {
            this.applyPresetPersona(personaId);
        }
    }

    showCustomView() {
        this.viewMain.classList.remove('active');
        this.viewCustom.classList.add('active');
        this.step2.classList.remove('active');
        this.step1.classList.add('active');
    }

    showMainView() {
        this.viewCustom.classList.remove('active');
        this.viewMain.classList.add('active');
    }

    nextStep() {
        const nameInput = document.getElementById('custom-name');
        if (!nameInput.value.trim()) {
            alert('유저 네임을 입력해주세요.');
            nameInput.focus();
            return;
        }
        this.step1.classList.remove('active');
        this.step2.classList.add('active');
    }

    prevStep() {
        this.step2.classList.remove('active');
        this.step1.classList.add('active');
    }

    async applyPresetPersona(personaId) {
        try {
            const response = await fetch(this.apiUrlSelect, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ persona_id: personaId })
            });
            const result = await response.json();
            if (result.success) {
                this.showStatus(`✅ '${result.profile.persona_name}' 적용 완료!`, 'success');
                this.updatePreview(result.profile);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error(error);
            this.showStatus('❌ 적용 실패', 'error');
        }
    }

    // [핵심 수정] 커스텀 프로필 생성 로직
    async generateAndSaveCustomProfile() {
        try {
            // 1. 기본 정보
            const name = document.getElementById('custom-name').value.trim();
            const ageGroup = document.getElementById('custom-age-group').value;
            const gender = document.getElementById('custom-gender').value;
            const occupation = document.getElementById('custom-occupation').value;

            let age = 25;
            if (ageGroup === '30s') age = 35;
            else if (ageGroup === '40s') age = 45;
            else if (ageGroup === '50s') age = 55;

            // 2. 속성 수집 (선택형)
            let selectedAttributes = [];

            // 2-1. 라디오 버튼 양자택일 처리
            // 빈 문자열("") 값은 추가되지 않음 (예: 건성, 모른다, 화려함, 해당없음)
            const radioNames = [
                'drink_temp', 'drink_sweet', 'drink_caff', 'drink_freq',
                'skin_type', 'makeup_style', 'exercise_type', 'alcohol_limit'
            ];

            radioNames.forEach(name => {
                const checked = document.querySelector(`input[name="${name}"]:checked`);
                if (checked && checked.value) {
                    selectedAttributes.push(checked.value);
                }
            });

            // 2-2. 다중 선택 체크박스 처리
            const checkboxes = document.querySelectorAll('#custom-profile-form input[type="checkbox"]:checked');
            checkboxes.forEach(cb => {
                if (cb.value) selectedAttributes.push(cb.value);
                const group = cb.getAttribute('data-group');
                if (group && this.groupMapping[group]) {
                    selectedAttributes.push(...this.groupMapping[group]);
                }
            });

            // 3. 직업 자동 추가
            if (occupation === 'student') selectedAttributes.push('capstone');
            if (occupation === 'worker') selectedAttributes.push('overtime');

            // 4. 랜덤 속성 배정 (50% 확률)
            this.randomAttributesPool.forEach(attr => {
                if (Math.random() < 0.5) selectedAttributes.push(attr);
            });
            
            selectedAttributes = [...new Set(selectedAttributes)];

            // 5. 프로필 객체 생성
            const profile = {
                user_id: "custom",
                persona_type: "custom",
                persona_name: name,
                age_group: ageGroup,
                age: age,
                gender: gender,
                occupation: [occupation],
                living_type: [],
                allergies: [],
                vegan: false,
                attribute_preferences: selectedAttributes,
                context: {
                    current_time: "evening",
                    day_type: "weekday",
                    weather: "sunny",
                    season: "winter"
                },
                personalization_level: 'high'
            };

            // 6. 서버 전송
            const response = await fetch(this.apiUrlUpdate, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(profile)
            });
            const data = await response.json();

            if (data.success) {
                this.showStatus('✅ 나만의 커스텀 프로필이 생성되었습니다!', 'success');
                this.updatePreview(profile);
                setTimeout(() => this.showMainView(), 1500);
            } else {
                throw new Error(data.error);
            }

        } catch (error) {
            console.error(error);
            this.showStatus('❌ 생성 실패: ' + error.message, 'error');
        }
    }

    async loadCurrentProfile() {
        try {
            const res = await fetch(this.apiUrlProfile);
            const data = await res.json();
            if (data.success) this.updatePreview(data.profile);
        } catch (e) { console.error(e); }
    }

    showStatus(msg, type) {
        this.statusMessage.textContent = msg;
        this.statusMessage.className = `status-message ${type}`;
        this.statusMessage.style.display = 'block';
        setTimeout(() => { this.statusMessage.style.display = 'none'; }, 3000);
    }

    updatePreview(profile) {
        this.profilePreview.textContent = JSON.stringify(profile, null, 2);
    }
}

window.adminPanel = new AdminPanel();