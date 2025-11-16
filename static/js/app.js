/**
 * 스마트 광고 AR 시스템 - 프론트엔드 JavaScript
 *
 * 웹캠 스트림 → 프레임 캡처 → OCR 처리 → 오버레이 렌더링
 */

class SmartAdARSystem {
    constructor() {
        // DOM 요소
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlayLayer = document.getElementById('overlay-layer');
        this.loadingIndicator = document.getElementById('loading');
        this.cameraSelect = document.getElementById('camera-select');

        // 페르소나 표시 요소
        this.personaName = document.getElementById('persona-name');
        this.personaTags = document.getElementById('persona-tags');

        // 버튼
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.toggleOcrBtn = document.getElementById('toggle-ocr');

        // 통계 요소
        this.fpsElement = document.getElementById('fps');
        this.ocrCountElement = document.getElementById('ocr-count');
        this.overlayCountElement = document.getElementById('overlay-count');
        this.debugInfo = document.getElementById('debug-info');
        this.ocrResults = document.getElementById('ocr-results');

        // 상태
        this.stream = null;
        this.isRunning = false;
        this.ocrEnabled = true;
        this.isProcessing = false;
        this.cameras = [];
        this.selectedDeviceId = null;

        // 통계
        this.frameCount = 0;
        this.ocrCount = 0;
        this.lastFpsTime = Date.now();
        this.fps = 0;

        // 설정
        this.ocrInterval = 20; // N프레임마다 OCR 실행 (CPU 부하 감소)
        this.frameCounter = 0;

        // API 엔드포인트 (동적으로 현재 프로토콜 사용)
        const protocol = window.location.protocol; // http: 또는 https:
        const host = window.location.host; // hostname:port
        this.apiUrl = `${protocol}//${host}/api/process_frame`;
        this.profileUrl = `${protocol}//${host}/api/user_profile`;

        this.init();
    }

    async init() {
        // 이벤트 리스너 등록
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.toggleOcrBtn.addEventListener('click', () => this.toggleOCR());
        this.cameraSelect.addEventListener('change', (e) => {
            this.selectedDeviceId = e.target.value;
            this.updateDebugInfo(`카메라 선택: ${e.target.options[e.target.selectedIndex].text}`);
        });

        // 사용자 프로필 로드
        await this.loadUserProfile();

        // 카메라 목록 로드
        await this.loadCameras();

        this.updateDebugInfo('시스템 초기화 완료');
    }

    async loadCameras() {
        try {
            this.updateDebugInfo('카메라 권한 요청 중...');

            // getUserMedia 사용 가능 여부 확인
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('이 브라우저는 getUserMedia를 지원하지 않습니다. HTTPS로 접속했는지 확인하세요.');
            }

            // 카메라 권한 요청 (목록을 얻기 위해 필요)
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());

            this.updateDebugInfo('카메라 목록 불러오는 중...');

            // 카메라 목록 가져오기
            const devices = await navigator.mediaDevices.enumerateDevices();
            this.cameras = devices.filter(device => device.kind === 'videoinput');

            this.updateDebugInfo(`총 ${this.cameras.length}개의 카메라 발견`);

            // 드롭다운 채우기
            this.cameraSelect.innerHTML = '';
            this.cameras.forEach((camera, index) => {
                const option = document.createElement('option');
                option.value = camera.deviceId;

                // 카메라 이름 (Camo는 "Camo" 문자열 포함)
                let label = camera.label || `카메라 ${index + 1}`;
                option.textContent = label;

                this.cameraSelect.appendChild(option);

                this.updateDebugInfo(`카메라 ${index + 1}: ${label}`);

                // Camo 카메라를 기본으로 선택
                if (label.toLowerCase().includes('camo') || label.toLowerCase().includes('reincubate')) {
                    option.selected = true;
                    this.selectedDeviceId = camera.deviceId;
                    this.updateDebugInfo(`✓ Camo 카메라 자동 선택: ${label}`);
                }
            });

            // 선택된 카메라가 없으면 첫 번째 카메라 선택
            if (!this.selectedDeviceId && this.cameras.length > 0) {
                this.selectedDeviceId = this.cameras[0].deviceId;
                this.updateDebugInfo(`기본 카메라 선택됨`);
            }

            this.updateDebugInfo(`✓ 카메라 로드 완료`);

        } catch (error) {
            console.error('카메라 목록 로드 오류:', error);
            const errorMsg = `카메라 로드 실패: ${error.name} - ${error.message}`;
            this.updateDebugInfo(errorMsg);
            this.cameraSelect.innerHTML = '<option>카메라 로드 실패</option>';
            alert(`카메라 접근 오류:\n${error.message}\n\nHTTPS로 접속했는지 확인하세요.`);
        }
    }

    async startCamera() {
        try {
            this.updateDebugInfo('카메라 접근 중...');

            // 선택된 카메라 확인
            if (!this.selectedDeviceId) {
                alert('카메라를 선택해주세요.');
                return;
            }

            // 웹캠 스트림 요청 (선택한 카메라 사용)
            const constraints = {
                video: {
                    deviceId: { exact: this.selectedDeviceId },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);

            this.video.srcObject = this.stream;
            this.isRunning = true;

            // 버튼 상태 업데이트
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.cameraSelect.disabled = true;

            const selectedCamera = this.cameraSelect.options[this.cameraSelect.selectedIndex].text;
            this.updateDebugInfo(`카메라 시작: ${selectedCamera}`);

            // 비디오 메타데이터 로드 완료 시 캔버스 크기 설정
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.updateDebugInfo(`해상도: ${this.video.videoWidth}x${this.video.videoHeight}`);
                this.processFrame();
            });

        } catch (error) {
            console.error('카메라 접근 오류:', error);
            this.updateDebugInfo(`오류: ${error.message}`);
            alert('선택한 카메라에 접근할 수 없습니다. 다른 카메라를 선택하거나 권한을 확인하세요.');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.isRunning = false;
        this.video.srcObject = null;

        // 버튼 상태 업데이트
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.cameraSelect.disabled = false;

        // 오버레이 제거
        this.clearOverlays();

        this.updateDebugInfo('카메라 중지');
    }

    toggleOCR() {
        this.ocrEnabled = !this.ocrEnabled;
        this.toggleOcrBtn.textContent = this.ocrEnabled ? 'OCR ON' : 'OCR OFF';
        this.toggleOcrBtn.classList.toggle('off', !this.ocrEnabled);

        if (!this.ocrEnabled) {
            this.clearOverlays();
        }

        this.updateDebugInfo(`OCR ${this.ocrEnabled ? '활성화' : '비활성화'}`);
    }

    async processFrame() {
        if (!this.isRunning) return;

        // FPS 계산
        this.frameCount++;
        const now = Date.now();
        if (now - this.lastFpsTime >= 1000) {
            this.fps = this.frameCount;
            this.fpsElement.textContent = this.fps;
            this.frameCount = 0;
            this.lastFpsTime = now;
        }

        // OCR 처리 (N프레임마다 1회)
        this.frameCounter++;
        if (this.ocrEnabled && this.frameCounter >= this.ocrInterval && !this.isProcessing) {
            this.frameCounter = 0;
            await this.runOCR();
        }

        // 다음 프레임 요청
        requestAnimationFrame(() => this.processFrame());
    }

    async runOCR() {
        this.isProcessing = true;
        this.showLoading(true);

        try {
            // 캔버스에 현재 프레임 그리기
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

            // Base64로 인코딩
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);

            // 백엔드로 전송
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                this.ocrCount++;
                this.ocrCountElement.textContent = this.ocrCount;

                // OCR 결과 표시
                this.displayOCRResults(data.ocr_results);

                // 오버레이 렌더링
                this.renderOverlays(data.overlays);

                this.updateDebugInfo(`OCR 완료: ${data.ocr_results.length}개 텍스트 인식`);
            } else {
                throw new Error(data.error);
            }

        } catch (error) {
            console.error('OCR 처리 오류:', error);
            this.updateDebugInfo(`OCR 오류: ${error.message}`);
        } finally {
            this.isProcessing = false;
            this.showLoading(false);
        }
    }

    displayOCRResults(results) {
        if (!results || results.length === 0) {
            this.ocrResults.innerHTML = '<p>텍스트 인식 없음</p>';
            return;
        }

        let html = '';
        results.forEach((result, idx) => {
            html += `<p><strong>[${idx + 1}]</strong> ${result.text} (신뢰도: ${(result.confidence * 100).toFixed(1)}%)</p>`;
        });

        this.ocrResults.innerHTML = html;
    }

    renderOverlays(overlays) {
        // 기존 오버레이 제거
        this.clearOverlays();

        if (!overlays || overlays.length === 0) {
            return;
        }

        this.overlayCountElement.textContent = overlays.length;

        overlays.forEach(overlay => {
            const element = this.createOverlayElement(overlay);
            this.overlayLayer.appendChild(element);
        });
    }

    createOverlayElement(overlay) {
        const div = document.createElement('div');
        div.className = `overlay-${overlay.type} ${overlay.position || ''} ${overlay.style || ''}`;
        div.textContent = overlay.content;

        // 커스텀 위치 (bbox 기반)
        if (overlay.bbox) {
            // bbox는 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형식
            const points = overlay.bbox;
            const x = Math.min(...points.map(p => p[0]));
            const y = Math.min(...points.map(p => p[1]));
            const width = Math.max(...points.map(p => p[0])) - x;
            const height = Math.max(...points.map(p => p[1])) - y;

            div.style.left = `${x}px`;
            div.style.top = `${y}px`;
            div.style.width = `${width}px`;
            div.style.height = `${height}px`;
        }

        // 애니메이션 효과
        div.style.animation = this.getAnimationForType(overlay.type);

        return div;
    }

    getAnimationForType(type) {
        const animations = {
            'promotion': 'slideInDown 0.5s ease-out, pulse 2s infinite',
            'badge': 'bounceIn 0.6s ease-out',
            'suggestion': 'fadeInUp 0.5s ease-out',
            'highlight': 'highlightPulse 1.5s infinite',
            'warning': 'shake 0.5s ease-out, blinkWarning 1s infinite'
        };

        return animations[type] || 'fadeIn 0.5s ease-out';
    }

    clearOverlays() {
        this.overlayLayer.innerHTML = '';
        this.overlayCountElement.textContent = '0';
    }

    showLoading(show) {
        if (show) {
            this.loadingIndicator.classList.add('active');
        } else {
            this.loadingIndicator.classList.remove('active');
        }
    }

    async loadUserProfile() {
        try {
            const response = await fetch(this.profileUrl);
            const data = await response.json();

            if (data.success) {
                this.updatePersonaDisplay(data.profile);
                this.updateDebugInfo('사용자 프로필 로드 완료');
            }
        } catch (error) {
            console.error('프로필 로드 오류:', error);
            this.personaName.textContent = '프로필 로드 실패';
            this.personaTags.innerHTML = '';
        }
    }

    updatePersonaDisplay(profile) {
        const prefs = profile.preferences || {};

        // 페르소나 이름 결정
        let personaName = '커스텀';
        const coffeeTypes = prefs.coffee_type || [];
        const interests = prefs.interests || [];

        // 간단한 페르소나 추론 (실제로는 관리 페이지에서 설정한 값을 사용해야 함)
        if (coffeeTypes.includes('라떼') && interests.includes('패션')) {
            personaName = '20대 여성';
        } else if (coffeeTypes.includes('아메리카노') && interests.includes('게임')) {
            personaName = '20대 남성';
        } else if (interests.includes('육아')) {
            personaName = '30대 여성';
        } else if (interests.includes('자동차')) {
            personaName = '30대 남성';
        } else if (interests.includes('건강') && interests.includes('여행')) {
            personaName = '시니어';
        }

        this.personaName.textContent = personaName;

        // 태그 생성
        const tags = [];

        // 선호 음료
        if (coffeeTypes.length > 0) {
            tags.push(...coffeeTypes.slice(0, 3));
        }

        // 관심사
        if (interests.length > 0) {
            tags.push(...interests.slice(0, 3));
        }

        // 알레르기
        const allergies = prefs.dietary?.allergies || [];
        if (allergies.length > 0) {
            tags.push(`⚠️ ${allergies.join(', ')} 알레르기`);
        }

        // 식이 제한
        if (prefs.dietary?.vegetarian) {
            tags.push('🥗 채식주의');
        }
        if (prefs.dietary?.vegan) {
            tags.push('🌱 비건');
        }

        // 가격 민감도
        const priceSens = prefs.price_sensitivity;
        if (priceSens === 'low') {
            tags.push('💎 프리미엄 선호');
        } else if (priceSens === 'high') {
            tags.push('💰 가성비 중시');
        }

        // 태그 HTML 생성
        this.personaTags.innerHTML = tags.map(tag =>
            `<span class="persona-tag">${tag}</span>`
        ).join('');
    }

    updateDebugInfo(message) {
        const timestamp = new Date().toLocaleTimeString();
        this.debugInfo.innerHTML = `<p>[${timestamp}] ${message}</p>` + this.debugInfo.innerHTML;

        // 최대 10개 메시지만 유지
        const messages = this.debugInfo.querySelectorAll('p');
        if (messages.length > 10) {
            messages[messages.length - 1].remove();
        }
    }
}

// 페이지 로드 완료 시 시스템 초기화
document.addEventListener('DOMContentLoaded', () => {
    const system = new SmartAdARSystem();
    console.log('스마트 광고 AR 시스템 초기화 완료');
});
