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
        this.readyUrl = `${protocol}//${host}/api/ready`;

        // 로딩 오버레이 요소
        this.backendLoading = document.getElementById('backend-loading');
        this.loadingProgressBar = document.getElementById('loading-progress-bar');
        this.ocrIcon = document.getElementById('ocr-icon');
        this.arucoIcon = document.getElementById('aruco-icon');
        this.personalizationIcon = document.getElementById('personalization-icon');
        this.loadingTitle = document.getElementById('loading-title');
        this.loadingMessage = document.getElementById('loading-message');

        this.init();
    }

    async init() {
        // 백엔드 준비 상태 확인
        await this.waitForBackendReady();

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

    async waitForBackendReady() {
        let progress = 0;
        const maxRetries = 60; // 최대 60초 대기
        let retries = 0;

        while (retries < maxRetries) {
            try {
                const response = await fetch(this.readyUrl);
                const data = await response.json();

                // 진행률 업데이트
                progress = 0;
                if (data.status.ocr_ready) {
                    progress += 33.3;
                    this.ocrIcon.textContent = '✅';
                }
                if (data.status.aruco_ready) {
                    progress += 33.3;
                    this.arucoIcon.textContent = '✅';
                }
                if (data.status.personalization_ready) {
                    progress += 33.4;
                    this.personalizationIcon.textContent = '✅';
                }

                this.loadingProgressBar.style.width = `${progress}%`;

                // 모든 컴포넌트가 준비되면
                if (data.ready) {
                    this.loadingTitle.textContent = '준비 완료!';
                    this.loadingMessage.textContent = '카메라를 시작할 수 있습니다';

                    // 0.5초 후 로딩 화면 제거
                    setTimeout(() => {
                        this.backendLoading.classList.add('fade-out');
                        setTimeout(() => {
                            this.backendLoading.style.display = 'none';
                        }, 500);
                    }, 500);

                    return;
                }

                // 1초 대기 후 재시도
                await new Promise(resolve => setTimeout(resolve, 1000));
                retries++;

            } catch (error) {
                console.error('백엔드 상태 확인 실패:', error);
                this.loadingMessage.textContent = '서버 연결 중... (' + (retries + 1) + '/' + maxRetries + ')';

                // 1초 대기 후 재시도
                await new Promise(resolve => setTimeout(resolve, 1000));
                retries++;
            }
        }

        // 타임아웃 시 경고 표시
        this.loadingTitle.textContent = '연결 실패';
        this.loadingMessage.textContent = '서버에 연결할 수 없습니다. 페이지를 새로고침하세요.';
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
            console.log('오버레이 없음');
            return;
        }

        console.log(`🎨 ${overlays.length}개 오버레이 렌더링 시작`);
        overlays.forEach((overlay, idx) => {
            console.log(`  오버레이 ${idx+1}:`, overlay);
        });

        this.overlayCountElement.textContent = overlays.length;

        overlays.forEach(overlay => {
            const element = this.createOverlayElement(overlay);
            this.overlayLayer.appendChild(element);
        });

        console.log(`✓ 오버레이 렌더링 완료`);
    }

    createOverlayElement(overlay) {
        const div = document.createElement('div');

        // position이 배열이면 CSS 클래스로 사용하지 않음
        const positionClass = Array.isArray(overlay.position) ? '' : (overlay.position || '');
        div.className = `overlay-${overlay.type} ${positionClass} ${overlay.style || ''}`.trim();

        // 텍스트 내용 설정 (줄바꿈 처리)
        if (overlay.content) {
            div.innerHTML = overlay.content.replace(/\n/g, '<br>');
        }

        // 위치 설정 우선순위: bbox > position 배열 > CSS 클래스
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
        } else if (Array.isArray(overlay.position) && overlay.position.length === 2) {
            // position이 [x, y] 배열 형태 - 비디오 크기 기준 상대 좌표로 변환
            const videoRect = this.video.getBoundingClientRect();
            const videoWidth = this.video.videoWidth || videoRect.width;
            const videoHeight = this.video.videoHeight || videoRect.height;

            // 원본 좌표를 퍼센트로 변환 (1920x1080 기준으로 가정)
            const baseWidth = 1920;
            const baseHeight = 1080;
            const xPercent = (overlay.position[0] / baseWidth) * 100;
            const yPercent = (overlay.position[1] / baseHeight) * 100;

            div.style.position = 'absolute';
            div.style.left = `${xPercent}%`;
            div.style.top = `${yPercent}%`;
            div.style.transform = 'translateX(0)'; // 기본값
        }

        // 색상 설정 (RGB 튜플) - 고급 글래스모피즘 스타일 (liquidGL 영감)
        if (overlay.color && Array.isArray(overlay.color) && overlay.color.length === 3) {
            const [r, g, b] = overlay.color;

            // 기본 배경 (투명도 높임 - 더 투명하게)
            div.style.backgroundColor = `rgba(${r}, ${g}, ${b}, 0.25)`;

            // 강화된 블러와 채도, 밝기 효과
            div.style.backdropFilter = 'blur(24px) saturate(200%) brightness(1.1)';
            div.style.webkitBackdropFilter = 'blur(24px) saturate(200%) brightness(1.1)';

            // 텍스트 색상 및 그림자
            div.style.color = r + g + b > 400 ? '#1a1a1a' : '#fff';
            div.style.textShadow = r + g + b > 400 ? '0 1px 2px rgba(255, 255, 255, 0.5)' : '0 1px 2px rgba(0, 0, 0, 0.3)';

            // 패딩 및 레이아웃
            div.style.padding = '14px 26px';
            div.style.borderRadius = '24px';

            // 다층 그림자로 깊이감 표현
            div.style.boxShadow = `
                0 8px 32px rgba(${r}, ${g}, ${b}, 0.15),
                0 2px 8px rgba(${r}, ${g}, ${b}, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.7),
                inset 0 -1px 0 rgba(0, 0, 0, 0.05)
            `;

            // 투명 테두리 (pseudo-element에서 그라디언트 적용)
            div.style.border = '1.5px solid transparent';
            div.style.backgroundClip = 'padding-box';

            // 타이포그래피
            div.style.fontSize = '0.95rem';
            div.style.fontWeight = '500';
            div.style.maxWidth = '85%';
            div.style.wordWrap = 'break-word';
            div.style.lineHeight = '1.5';

            // 유리 반사 하이라이트 추가 (::before 효과)
            const highlight = document.createElement('div');
            highlight.style.position = 'absolute';
            highlight.style.top = '0';
            highlight.style.left = '0';
            highlight.style.right = '0';
            highlight.style.height = '50%';
            highlight.style.background = 'linear-gradient(180deg, rgba(255, 255, 255, 0.35) 0%, rgba(255, 255, 255, 0) 100%)';
            highlight.style.borderRadius = '24px 24px 0 0';
            highlight.style.pointerEvents = 'none';
            div.appendChild(highlight);

            // 유리 테두리 굴절 효과 추가 (::after 효과)
            const border = document.createElement('div');
            border.style.position = 'absolute';
            border.style.inset = '-1.5px';
            border.style.borderRadius = '24px';
            border.style.padding = '1.5px';
            border.style.background = `linear-gradient(
                135deg,
                rgba(255, 255, 255, 0.7) 0%,
                rgba(255, 255, 255, 0.15) 50%,
                rgba(255, 255, 255, 0.5) 100%
            )`;
            border.style.webkitMask = 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)';
            border.style.webkitMaskComposite = 'xor';
            border.style.maskComposite = 'exclude';
            border.style.pointerEvents = 'none';
            border.style.zIndex = '-1';
            div.appendChild(border);
        }

        // 애니메이션 효과
        div.style.animation = this.getAnimationForType(overlay.type);

        return div;
    }

    getAnimationForType(type) {
        const animations = {
            'promotion': 'fadeIn 0.4s ease-out',
            'badge': 'fadeIn 0.3s ease-out',
            'suggestion': 'fadeIn 0.4s ease-out',
            'recommendation': 'fadeIn 0.4s ease-out',
            'highlight': 'subtlePulse 2s infinite',
            'warning': 'fadeIn 0.4s ease-out',
            'info': 'fadeIn 0.4s ease-out',
            'event': 'fadeIn 0.4s ease-out'
        };

        return animations[type] || 'fadeIn 0.4s ease-out';
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
        // 페르소나 이름 (저장된 값 사용)
        const personaName = profile.persona_name || '커스텀';
        this.personaName.textContent = personaName;

        // 태그 생성 (새로운 구조에 맞게)
        const tags = [];

        // 성별 + 나이
        if (profile.gender && profile.age) {
            const genderText = profile.gender === 'female' ? '여성' : profile.gender === 'male' ? '남성' : '기타';
            tags.push(`${genderText}, ${profile.age}세`);
        }

        // 직업
        if (profile.occupation && profile.occupation.length > 0) {
            tags.push(profile.occupation.join(', '));
        }

        // 선호 속성 (최대 3개)
        const attrPrefs = profile.attribute_preferences || [];
        if (attrPrefs.length > 0) {
            tags.push(...attrPrefs.slice(0, 3));
        }

        // 가격 민감도
        if (profile.price_sensitivity) {
            const priceText = {
                'low': '프리미엄 선호',
                'medium': '보통',
                'high': '가성비 중시'
            };
            tags.push(priceText[profile.price_sensitivity] || profile.price_sensitivity);
        }

        // 알레르기
        const allergies = profile.allergies || [];
        if (allergies.length > 0) {
            tags.push(`⚠️ ${allergies.join(', ')} 알레르기`);
        }

        // 식이 제한
        if (profile.vegan) {
            tags.push('🥗 비건');
        }
        if (profile.low_sugar_preference) {
            tags.push('🍬 저당 선호');
        }
        if (profile.low_caffeine_preference) {
            tags.push('☕ 저카페인 선호');
        }

        // 태그 HTML 생성 (최대 5개만 표시)
        this.personaTags.innerHTML = tags.slice(0, 5).map(tag =>
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
