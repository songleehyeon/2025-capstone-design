/**
 * 스마트 광고 AR 시스템 - 프론트엔드 JavaScript
 *
 * 웹캠 스트림 → 프레임 캡처 → OCR 처리 → 오버레이 렌더링
 */

/**
 * 스마트 광고 AR 시스템 - 프론트엔드 JavaScript (최종 통합 버전)
 * 기능: 웹캠 스트림, OCR 처리, 슬롯 기반 오버레이, 화면 초기화
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
        this.clearBtn = document.getElementById('clear-btn'); // [초기화 버튼]

        // 통계 및 디버그 요소
        this.fpsElement = document.getElementById('fps');
        this.ocrCountElement = document.getElementById('ocr-count');
        this.overlayCountElement = document.getElementById('overlay-count');
        this.debugInfo = document.getElementById('debug-info');
        this.ocrResults = document.getElementById('ocr-results');

        // 상태 변수
        this.stream = null;
        this.isRunning = false;
        this.ocrEnabled = true;
        this.isProcessing = false;
        this.cameras = [];
        this.selectedDeviceId = null;

        // 통계 변수
        this.frameCount = 0;
        this.ocrCount = 0;
        this.lastFpsTime = Date.now();
        this.fps = 0;

        // 설정
        this.ocrInterval = 2; // 2프레임마다 OCR 실행 (매우 빠른 반응, 마커 즉시 감지)
        this.frameCounter = 0;

        // API 엔드포인트
        const protocol = window.location.protocol;
        const host = window.location.host;
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
        // 백엔드 준비 대기
        await this.waitForBackendReady();

        // 이벤트 리스너 등록
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.toggleOcrBtn.addEventListener('click', () => this.toggleOCR());
        
        this.cameraSelect.addEventListener('change', (e) => {
            this.selectedDeviceId = e.target.value;
            this.updateDebugInfo(`카메라 선택: ${e.target.options[e.target.selectedIndex].text}`);
        });

        // [초기화 버튼 리스너]
        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => this.flashScreen());
        }

        // 초기 데이터 로드
        await this.loadUserProfile();
        await this.loadCameras();

        this.updateDebugInfo('시스템 초기화 완료');

    }

    // [신규 기능] 화면 및 OCR 결과 초기화 (OCR도 일시정지)
    async flashScreen() {
        // 1. OCR 기능 끄기 (재인식 방지)
        if (this.ocrEnabled) {
            this.toggleOCR(); // OCR OFF로 전환
        }

        // 2. 화면의 오버레이 제거
        this.clearOverlays();

        // 3. OCR 결과 텍스트 초기화
        if (this.ocrResults) {
            this.ocrResults.innerHTML = '<p>초기화 중...</p>';
        }

        // 4. 백엔드 OCR 캐시 초기화 (API 호출)
        try {
            const protocol = window.location.protocol;
            const host = window.location.host;
            const clearCacheUrl = `${protocol}//${host}/api/clear_cache`;

            const response = await fetch(clearCacheUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (data.success) {
                console.log('✓ 백엔드 OCR 캐시 초기화됨');
                if (this.ocrResults) {
                    this.ocrResults.innerHTML = '<p>초기화 완료 (OCR 대기중)</p>';
                }
                this.updateDebugInfo('🧹 화면 초기화 완료 (백엔드 캐시 포함)');
            } else {
                console.error('캐시 초기화 실패:', data.error);
                if (this.ocrResults) {
                    this.ocrResults.innerHTML = '<p>초기화 실패 - 다시 시도하세요</p>';
                }
            }
        } catch (error) {
            console.error('백엔드 캐시 초기화 오류:', error);
            if (this.ocrResults) {
                this.ocrResults.innerHTML = '<p>초기화 오류 발생</p>';
            }
            this.updateDebugInfo('⚠️ 캐시 초기화 오류 (프론트엔드만 초기화됨)');
        }
    }

    async waitForBackendReady() {
        let progress = 0;
        const maxRetries = 60;
        let retries = 0;

        while (retries < maxRetries) {
            try {
                const response = await fetch(this.readyUrl);
                const data = await response.json();

                progress = 0;
                if (data.status.ocr_ready) { progress += 33.3; this.ocrIcon.textContent = '✅'; }
                if (data.status.aruco_ready) { progress += 33.3; this.arucoIcon.textContent = '✅'; }
                if (data.status.personalization_ready) { progress += 33.4; this.personalizationIcon.textContent = '✅'; }

                this.loadingProgressBar.style.width = `${progress}%`;

                if (data.ready) {
                    this.loadingTitle.textContent = '준비 완료!';
                    this.loadingMessage.textContent = '카메라를 시작할 수 있습니다';
                    setTimeout(() => {
                        this.backendLoading.classList.add('fade-out');
                        setTimeout(() => { this.backendLoading.style.display = 'none'; }, 500);
                    }, 500);
                    return;
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
                retries++;
            } catch (error) {
                console.error('백엔드 상태 확인 실패:', error);
                this.loadingMessage.textContent = `서버 연결 중... (${retries + 1}/${maxRetries})`;
                await new Promise(resolve => setTimeout(resolve, 1000));
                retries++;
            }
        }
        this.loadingTitle.textContent = '연결 실패';
        this.loadingMessage.textContent = '서버에 연결할 수 없습니다. 페이지를 새로고침하세요.';
    }

    async loadCameras() {
        try {
            this.updateDebugInfo('카메라 권한 요청 중...');
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('이 브라우저는 getUserMedia를 지원하지 않습니다. HTTPS로 접속했는지 확인하세요.');
            }
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());

            this.updateDebugInfo('카메라 목록 불러오는 중...');
            const devices = await navigator.mediaDevices.enumerateDevices();
            this.cameras = devices.filter(device => device.kind === 'videoinput');

            this.cameraSelect.innerHTML = '';
            this.cameras.forEach((camera, index) => {
                const option = document.createElement('option');
                option.value = camera.deviceId;
                let label = camera.label || `카메라 ${index + 1}`;
                option.textContent = label;
                this.cameraSelect.appendChild(option);

                if (label.toLowerCase().includes('camo') || label.toLowerCase().includes('reincubate')) {
                    option.selected = true;
                    this.selectedDeviceId = camera.deviceId;
                }
            });

            if (!this.selectedDeviceId && this.cameras.length > 0) {
                this.selectedDeviceId = this.cameras[0].deviceId;
            }
            this.updateDebugInfo(`✓ 카메라 로드 완료 (${this.cameras.length}개)`);
        } catch (error) {
            console.error('카메라 로드 오류:', error);
            this.updateDebugInfo(`카메라 오류: ${error.message}`);
            alert(`카메라 접근 오류:\n${error.message}`);
        }
    }

    async startCamera() {
        try {
            if (!this.selectedDeviceId) { alert('카메라를 선택해주세요.'); return; }
            
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

            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.cameraSelect.disabled = true;

            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.updateDebugInfo(`해상도: ${this.video.videoWidth}x${this.video.videoHeight}`);
                this.processFrame();
            });
        } catch (error) {
            console.error('카메라 시작 오류:', error);
            alert(`카메라 시작 실패: ${error.message}`);
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.isRunning = false;
        this.video.srcObject = null;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.cameraSelect.disabled = false;
        this.clearOverlays();
        this.updateDebugInfo('카메라 중지');
    }

    toggleOCR() {
        this.ocrEnabled = !this.ocrEnabled;
        this.toggleOcrBtn.textContent = this.ocrEnabled ? 'OCR ON' : 'OCR OFF';
        this.toggleOcrBtn.classList.toggle('off', !this.ocrEnabled);
        
        if (!this.ocrEnabled) {
            // OCR을 끌 때는 오버레이도 같이 지워주는 것이 자연스럽습니다.
            // this.clearOverlays(); 
        }
        this.updateDebugInfo(`OCR ${this.ocrEnabled ? '활성화' : '비활성화'}`);
    }

    async processFrame() {
        if (!this.isRunning) return;

        this.frameCount++;
        const now = Date.now();
        if (now - this.lastFpsTime >= 1000) {
            this.fps = this.frameCount;
            if (this.fpsElement) {
                this.fpsElement.textContent = this.fps;
            }
            this.frameCount = 0;
            this.lastFpsTime = now;
        }

        this.frameCounter++;
        if (this.ocrEnabled && this.frameCounter >= this.ocrInterval && !this.isProcessing) {
            this.frameCounter = 0;
            await this.runOCR();
        }

        requestAnimationFrame(() => this.processFrame());
    }

    async runOCR() {
        this.isProcessing = true;
        this.showLoading(true);

        try {
            console.log('🎬 OCR 시작...');
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);

            console.log('📤 서버로 요청 전송 중...');
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            console.log('📥 서버 응답 받음:', response.status);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            console.log('📦 응답 데이터:', data);

            if (data.success) {
                this.ocrCount++;
                if (this.ocrCountElement) {
                    this.ocrCountElement.textContent = this.ocrCount;
                }

                console.log('✓ OCR 결과:', data.ocr_results);
                console.log('✓ 오버레이 개수:', data.overlays ? data.overlays.length : 0);

                // OCR 결과 표시
                this.displayOCRResults(data.ocr_results);

                // 오버레이 렌더링 (빈 배열이면 자동으로 clearOverlays만 호출됨)
                this.renderOverlays(data.overlays);

                // 🔥 중요: OCR 결과가 비어있으면 즉시 오버레이 제거
                if (!data.ocr_results || data.ocr_results.length === 0) {
                    this.clearOverlays();
                    this.updateDebugInfo('⚠ 키워드 없음 - 오버레이 제거됨');
                } else {
                    this.updateDebugInfo(`OCR 완료: ${data.ocr_results.length}개 인식, ${data.overlays.length}개 오버레이`);
                }
            } else {
                console.error('❌ 서버 에러:', data.error);
                throw new Error(data.error);
            }
        } catch (error) {
            console.error('❌ OCR 오류:', error);
            this.updateDebugInfo(`❌ 오류 발생: ${error.message}`);
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
            html += `<p><strong>[${idx + 1}]</strong> ${result.text} (${(result.confidence * 100).toFixed(0)}%)</p>`;
        });
        this.ocrResults.innerHTML = html;
    }

    // [수정] 슬롯 기반 배치를 위해 idx 전달
    renderOverlays(overlays) {
        console.log('🎨 renderOverlays 호출됨, 오버레이 개수:', overlays ? overlays.length : 0);
        if (!overlays) overlays = [];

        const existingElements = Array.from(this.overlayLayer.children);
        console.log('📋 기존 오버레이 요소 개수:', existingElements.length);

        // 1. 개수가 다르면 싹 지우고 새로 그립니다 (초기화 등 급격한 변화 시)
        if (existingElements.length !== overlays.length) {
            console.log('🔄 개수가 달라서 재생성:', existingElements.length, '→', overlays.length);
            this.clearOverlays();
            overlays.forEach((overlay, idx) => {
                console.log(`  생성 중 [${idx}]:`, overlay.content?.substring(0, 30));
                const element = this.createOverlayElement(overlay, idx);
                this.overlayLayer.appendChild(element);
                console.log(`  ✓ 요소 추가됨 [${idx}]`, element);
            });
            console.log('✓ overlayLayer에 추가된 자식 요소 수:', this.overlayLayer.children.length);
            return;
        }

        // 2. 개수가 같으면 "위치와 내용만" 부드럽게 업데이트 (핵심!)
        console.log('🔄 개수가 같아서 업데이트만 수행');
        overlays.forEach((overlay, idx) => {
            const div = existingElements[idx];

            // 내용이 다를 때만 업데이트 (깜빡임 방지)
            const newContent = overlay.content ? overlay.content : '';
            if (div.textContent !== newContent) {
                div.textContent = newContent;
                console.log(`  업데이트 [${idx}]:`, newContent.substring(0, 30));
            }

            // 목표 위치 계산
            let targetLeft, targetTop;
            if (overlay.bbox) {
                const points = overlay.bbox;
                const x = Math.min(...points.map(p => p[0]));
                const y = Math.min(...points.map(p => p[1]));
                targetLeft = `${x}px`;
                targetTop = `${y}px`;
            } else {
                // 슬롯 위치 (BBox 없을 때)
                const slots = [[5, 10], [60, 10], [10, 35], [60, 35], [5, 60], [60, 60]];
                const slot = slots[idx % slots.length];
                targetLeft = `${slot[0]}%`;
                targetTop = `${slot[1]}%`;
            }

            // 위치만 변경 -> CSS transition이 동작하여 스르륵 움직임
            div.style.left = targetLeft;
            div.style.top = targetTop;
        });

        if(this.overlayCountElement) this.overlayCountElement.textContent = overlays.length;
    }

    // [수정] 슬롯 기반 위치 + 부드러운 바운스 애니메이션
    createOverlayElement(overlay, idx) {
        console.log(`🏗️ createOverlayElement [${idx}] 시작`, overlay);
        const div = document.createElement('div');
        div.className = `overlay-card`;

        // 내용 삽입
        if (overlay.content) {
            div.textContent = overlay.content;
            console.log(`  ✓ 내용 설정: "${overlay.content.substring(0, 30)}..."`);
        } else {
            console.warn(`  ⚠️ overlay.content가 없음!`);
        }

        // 위치 설정 (bbox 우선, 없으면 슬롯)
        if (overlay.bbox) {
            const points = overlay.bbox;
            const x = Math.min(...points.map(p => p[0]));
            const y = Math.min(...points.map(p => p[1]));
            div.style.left = `${x}px`;
            div.style.top = `${y}px`;
            console.log(`  ✓ BBox 위치: (${x}px, ${y}px)`);
        } else {
            // 중앙(30~70%)을 피하는 좌우 슬롯
            const slots = [
                [5, 10], [60, 10],  // 상단 좌우
                [10, 35], [60, 35], // 중단 좌우
                [5, 60], [60, 60]   // 하단 좌우
            ];
            const slot = slots[idx % slots.length];
            div.style.position = 'absolute';
            div.style.left = `${slot[0]}%`;
            div.style.top = `${slot[1]}%`;
            console.log(`  ✓ 슬롯 위치: (${slot[0]}%, ${slot[1]}%)`);
        }

        const duration = Math.random() * 0.3 + 0.9;
        const delay = Math.random() * 0.5;

        div.style.animationName = 'bounceFloat';
        div.style.animationDuration = `${duration}s`;
        div.style.animationDelay = `${delay}s`;
        div.style.animationIterationCount = 'infinite';
        div.style.animationTimingFunction = 'ease-in-out';
        div.style.animationDirection = 'alternate';

        console.log(`  ✓ 요소 생성 완료:`, div);
        return div;
    }

    clearOverlays() {
        this.overlayLayer.innerHTML = '';
        if (this.overlayCountElement) {
            this.overlayCountElement.textContent = '0';
        }
    }

    showLoading(show) {
        if (show) this.loadingIndicator.classList.add('active');
        else this.loadingIndicator.classList.remove('active');
    }

    async loadUserProfile() {
        try {
            const response = await fetch(this.profileUrl);
            const data = await response.json();
            if (data.success) {
                this.updatePersonaDisplay(data.profile);
                this.updateDebugInfo('프로필 로드 완료');
            }
        } catch (error) {
            console.error('프로필 로드 오류:', error);
        }
    }

    updatePersonaDisplay(profile) {
        const personaName = profile.persona_name || '커스텀';
        const gender = profile.gender === 'female' ? '여' : '남';
        const age = profile.age || '?';

        // admin 페이지처럼 간단하게 표시: "이름 (성별, 나이)"
        this.personaName.textContent = `${personaName} (${gender}, ${age}세)`;
    }

    updateDebugInfo(message) {
        const timestamp = new Date().toLocaleTimeString();
        this.debugInfo.innerHTML = `<p>[${timestamp}] ${message}</p>` + this.debugInfo.innerHTML;
        const messages = this.debugInfo.querySelectorAll('p');
        if (messages.length > 10) messages[messages.length - 1].remove();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const system = new SmartAdARSystem();
    console.log('스마트 광고 AR 시스템 초기화 완료');
});