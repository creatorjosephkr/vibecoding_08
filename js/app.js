/**
 * Main Application Controller (Enhanced Version)
 * UI 상호작용 및 고정밀 인식 파이프라인 제어
 */

document.addEventListener('DOMContentLoaded', async () => {
  // DOM Elements
  const canvasElement = document.getElementById('drawingCanvas');
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const uploadPreviewWrapper = document.getElementById('uploadPreviewWrapper');
  const imagePreview = document.getElementById('imagePreview');
  const removeImageBtn = document.getElementById('removeImageBtn');

  const tabDraw = document.getElementById('tabDraw');
  const tabCamera = document.getElementById('tabCamera');
  const tabUpload = document.getElementById('tabUpload');
  const canvasWrapper = document.getElementById('canvasWrapper');
  const canvasToolbar = document.getElementById('canvasToolbar');
  const canvasGuideHint = document.getElementById('canvasGuideHint');

  // Camera Elements
  const cameraViewWrapper = document.getElementById('cameraViewWrapper');
  const cameraStreamContainer = document.getElementById('cameraStreamContainer');
  const cameraVideo = document.getElementById('cameraVideo');
  const switchCameraBtn = document.getElementById('switchCameraBtn');
  const cameraCapturedContainer = document.getElementById('cameraCapturedContainer');
  const capturedImagePreview = document.getElementById('capturedImagePreview');
  const retakePhotoBtn = document.getElementById('retakePhotoBtn');
  const capturePhotoBtn = document.getElementById('capturePhotoBtn');
  const nativeCameraBtn = document.getElementById('nativeCameraBtn');
  const cameraFileInput = document.getElementById('cameraFileInput');

  // Scope & Mode Elements
  const scopeSingleBtn = document.getElementById('scopeSingleBtn');
  const scopeSentenceBtn = document.getElementById('scopeSentenceBtn');
  const minSizeGuideBox = document.getElementById('minSizeGuideBox');
  const sentenceGuideLines = document.getElementById('sentenceGuideLines');

  const modeDigitsBtn = document.getElementById('modeDigitsBtn');
  const modeLettersBtn = document.getElementById('modeLettersBtn');
  const modeAllBtn = document.getElementById('modeAllBtn');
  const modeButtons = [modeDigitsBtn, modeLettersBtn, modeAllBtn];

  // Engine Elements
  const engineGoogleBtn = document.getElementById('engineGoogleBtn');
  const engineLocalBtn = document.getElementById('engineLocalBtn');

  const strokeWidthSlider = document.getElementById('strokeWidth');
  const strokeWidthVal = document.getElementById('strokeWidthVal');
  const undoBtn = document.getElementById('undoBtn');
  const redoBtn = document.getElementById('redoBtn');
  const eraserBtn = document.getElementById('eraserBtn');
  const clearBtn = document.getElementById('clearBtn');

  const recognizeBtn = document.getElementById('recognizeBtn');
  const recognizeBtnText = document.getElementById('recognizeBtnText');
  const recognizeBtnSpinner = document.getElementById('recognizeBtnSpinner');
  const statusBadge = document.getElementById('statusBadge');

  const resultCharacter = document.getElementById('resultCharacter');
  const confidencePercent = document.getElementById('confidencePercent');
  const confidenceBarFill = document.getElementById('confidenceBarFill');
  const copyBtn = document.getElementById('copyBtn');

  const candidateList = document.getElementById('candidateList');
  const historyList = document.getElementById('historyList');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  const toastNotice = document.getElementById('toastNotice');
  const toastMessage = document.getElementById('toastMessage');

  // Threshold Elements
  const thresholdSlider = document.getElementById('thresholdSlider');
  const thresholdValue = document.getElementById('thresholdValue');
  const thresholdPresetBtns = document.querySelectorAll('.threshold-preset-btn');
  const thresholdAlertBox = document.getElementById('thresholdAlertBox');
  const thresholdAlertText = document.getElementById('thresholdAlertText');

  // Size Warning Modal Elements
  const sizeWarningModal = document.getElementById('sizeWarningModal');
  const modalRewriteBtn = document.getElementById('modalRewriteBtn');
  const modalProceedBtn = document.getElementById('modalProceedBtn');

  // QR Code Modal Elements
  const qrModalBtn = document.getElementById('qrModalBtn');
  const qrModal = document.getElementById('qrModal');
  const qrModalCloseBtn = document.getElementById('qrModalCloseBtn');
  const qrModalCloseXBtn = document.getElementById('qrModalCloseXBtn');

  const themeToggleBtn = document.getElementById('themeToggleBtn');

  // Instances
  const drawingCanvas = new DrawingCanvas('drawingCanvas');
  const recognizer = new HandwritingRecognizer();

  // App State
  let currentTab = 'draw'; // 'draw' | 'camera' | 'upload'
  let cameraStream = null;
  let currentFacingMode = 'environment';
  let capturedPhotoSource = null;
  let currentScope = 'single'; // 'single': 1개 문자 | 'sentence': 단어/문장
  let currentMode = 'digits'; // 'digits' | 'letters' | 'all'
  let currentEngine = 'google'; // 'google': 획 궤적 초고정밀 | 'local': Tesseract+CNN
  let currentThreshold = parseInt(localStorage.getItem('handwriting_threshold') || '70', 10);
  let uploadedImageSource = null;
  let recognitionHistory = JSON.parse(localStorage.getItem('handwriting_history') || '[]');
  let isEraserActive = false;

  // 1. 초기 히스토리 렌더링
  renderHistory();

  function updateStatusBadge(isProcessing = false) {
    if (!statusBadge) return;
    if (isProcessing) {
      statusBadge.textContent = '분석 중';
      statusBadge.className = 'status-badge processing';
      return;
    }

    const engineLabel = currentEngine === 'google' ? 'Google AI' : '로컬 AI';
    const scopeLabel = currentScope === 'single' ? '1개 문자' : '단어/문장';
    const modeLabel = currentMode === 'digits' ? '숫자' : currentMode === 'letters' ? '영문' : '통합';
    statusBadge.textContent = `준비 완료 (${engineLabel} · ${scopeLabel} · ${modeLabel})`;
    statusBadge.className = 'status-badge';
  }

  // 2. Tesseract AI 모델 사전 로드 (백그라운드)
  recognizer.init((status, progress) => {
    if (statusBadge && status !== 'ready') {
      const pct = Math.round(progress * 100);
      statusBadge.textContent = `AI 로딩 중 (${pct}%)`;
      statusBadge.classList.add('processing');
    }
  }).then(() => {
    updateStatusBadge();
  }).catch((e) => {
    console.warn('AI 엔진 준비 지연:', e);
    if (statusBadge) statusBadge.textContent = '대기 중';
  });

  // ==========================================
  // 인식 단위 선택 (1개 문자 모드 vs 단어/문장 모드)
  // ==========================================
  function setRecognitionScope(scope) {
    currentScope = scope;
    if (scopeSingleBtn) scopeSingleBtn.classList.toggle('active', scope === 'single');
    if (scopeSentenceBtn) scopeSentenceBtn.classList.toggle('active', scope === 'sentence');

    recognizer.setScope(scope);

    if (scope === 'single') {
      if (minSizeGuideBox) minSizeGuideBox.style.display = 'block';
      if (sentenceGuideLines) sentenceGuideLines.style.display = 'none';
      if (canvasGuideHint) {
        canvasGuideHint.textContent = '점선 상자보다 큼직하게 1글자를 작성해 주세요.';
      }
      showToast('[1개 문자 모드] 한 글자씩 정밀 판독합니다.');
    } else {
      if (minSizeGuideBox) minSizeGuideBox.style.display = 'none';
      if (sentenceGuideLines) sentenceGuideLines.style.display = 'flex';
      if (canvasGuideHint) {
        canvasGuideHint.textContent = '가로 방향으로 단어나 문장을 이어서 작성해 주세요.';
      }
      showToast('[단어/문장 모드] 여러 글자나 문장을 가로로 이어서 인식합니다.');
    }

    updateStatusBadge();
  }

  scopeSingleBtn?.addEventListener('click', () => setRecognitionScope('single'));
  scopeSentenceBtn?.addEventListener('click', () => setRecognitionScope('sentence'));

  // ==========================================
  // 인식 기준치(확신도 임계값) 설정 로직
  // ==========================================
  function updateThresholdUI(val) {
    currentThreshold = val;
    if (thresholdSlider) thresholdSlider.value = val;
    if (thresholdValue) thresholdValue.textContent = `${val}%`;
    localStorage.setItem('handwriting_threshold', `${val}`);

    thresholdPresetBtns.forEach((btn) => {
      const bVal = parseInt(btn.getAttribute('data-val'), 10);
      btn.classList.toggle('active', bVal === val);
    });
  }

  // 초기 기준치 적용
  updateThresholdUI(currentThreshold);

  thresholdSlider?.addEventListener('input', (e) => {
    updateThresholdUI(parseInt(e.target.value, 10));
  });

  thresholdPresetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const val = parseInt(btn.getAttribute('data-val'), 10);
      updateThresholdUI(val);
      showToast(`인식 기준치가 ${val}%로 설정되었습니다.`);
    });
  });

  // ==========================================
  // 인식 대상 모드 선택 (숫자 전용 / 영문 전용 / 전체)
  // ==========================================
  function setRecognitionMode(mode) {
    currentMode = mode;
    modeButtons.forEach((btn) => {
      if (btn) {
        btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
      }
    });

    recognizer.setMode(mode);

    if (mode === 'digits') {
      showToast('[숫자] 모드: 0~9 손글씨에 최적화되었습니다.');
    } else if (mode === 'letters') {
      showToast('[영문] 모드: A~Z 알파벳에 최적화되었습니다.');
    } else {
      showToast('[통합] 모드: 숫자와 영문을 함께 판별합니다.');
    }

    updateStatusBadge();
  }

  modeDigitsBtn?.addEventListener('click', () => setRecognitionMode('digits'));
  modeLettersBtn?.addEventListener('click', () => setRecognitionMode('letters'));
  modeAllBtn?.addEventListener('click', () => setRecognitionMode('all'));

  // ==========================================
  // AI 엔진 선택 (Google 손글씨 AI vs 로컬 AI)
  // ==========================================
  function setEngine(engine) {
    currentEngine = engine;
    if (engineGoogleBtn) engineGoogleBtn.classList.toggle('active', engine === 'google');
    if (engineLocalBtn) engineLocalBtn.classList.toggle('active', engine === 'local');

    if (engine === 'google') {
      showToast('[Google AI 엔진] 획 궤적 기반 99% 초고정밀 손글씨 인식을 사용합니다.');
    } else {
      showToast('[로컬 AI 엔진] 브라우저 내 Tesseract OCR + MNIST CNN 앙상블을 사용합니다.');
    }

    updateStatusBadge();
  }

  engineGoogleBtn?.addEventListener('click', () => setEngine('google'));
  engineLocalBtn?.addEventListener('click', () => setEngine('local'));

  // ==========================================
  // 탭 전환 로직 (직접 그리기 / 사진 촬영 / 파일 업로드)
  // ==========================================
  function switchTab(tab) {
    currentTab = tab;
    tabDraw.classList.toggle('active', tab === 'draw');
    tabCamera.classList.toggle('active', tab === 'camera');
    tabUpload.classList.toggle('active', tab === 'upload');

    // 1. 직접 그리기 (Canvas)
    if (tab === 'draw') {
      canvasWrapper.style.display = 'flex';
      canvasToolbar.style.display = 'flex';
      if (canvasGuideHint) canvasGuideHint.style.display = 'flex';
      setTimeout(() => drawingCanvas.handleResize(), 30);
    } else {
      canvasWrapper.style.display = 'none';
      canvasToolbar.style.display = 'none';
      if (canvasGuideHint) canvasGuideHint.style.display = 'none';
    }

    // 2. 사진 촬영 (Camera)
    if (tab === 'camera') {
      cameraViewWrapper.style.display = 'flex';
      if (!capturedPhotoSource) {
        startCameraStream();
      }
    } else {
      cameraViewWrapper.style.display = 'none';
      stopCameraStream();
    }

    // 3. 파일 업로드 (Upload)
    if (tab === 'upload') {
      if (uploadedImageSource && currentTab !== 'camera') {
        uploadZone.style.display = 'none';
        uploadPreviewWrapper.style.display = 'block';
      } else {
        uploadZone.style.display = 'flex';
        uploadPreviewWrapper.style.display = 'none';
      }
    } else {
      uploadZone.style.display = 'none';
      uploadPreviewWrapper.style.display = 'none';
    }
  }

  tabDraw.addEventListener('click', () => switchTab('draw'));
  tabCamera.addEventListener('click', () => switchTab('camera'));
  tabUpload.addEventListener('click', () => switchTab('upload'));

  // ==========================================
  // 카메라 사진 촬영 기능 로직
  // ==========================================
  async function startCameraStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast('브라우저 카메라 스트림이 지원되지 않습니다. [기기 카메라 열기]를 이용해 주세요.');
      return;
    }

    stopCameraStream();

    try {
      const constraints = {
        video: {
          facingMode: currentFacingMode,
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      };

      cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
      if (cameraVideo) {
        cameraVideo.srcObject = cameraStream;
        await cameraVideo.play();
      }

      cameraStreamContainer.style.display = 'flex';
      cameraCapturedContainer.style.display = 'none';
      capturePhotoBtn.style.display = 'inline-flex';
    } catch (err) {
      console.warn('카메라 스트림 시작 오류:', err);
      showToast('카메라 접근 권한이 필요합니다. [기기 카메라 열기] 버튼을 눌러보세요.');
    }
  }

  function stopCameraStream() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }
    if (cameraVideo) {
      cameraVideo.srcObject = null;
    }
  }

  function capturePhoto() {
    if (!cameraVideo || !cameraStream) {
      showToast('카메라 뷰파인더가 준비되지 않았습니다.');
      return;
    }

    const videoW = cameraVideo.videoWidth || 640;
    const videoH = cameraVideo.videoHeight || 480;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = videoW;
    tempCanvas.height = videoH;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0, videoW, videoH);

    const dataUrl = tempCanvas.toDataURL('image/png');
    capturedPhotoSource = dataUrl;
    uploadedImageSource = dataUrl;

    capturedImagePreview.src = dataUrl;
    cameraStreamContainer.style.display = 'none';
    cameraCapturedContainer.style.display = 'flex';
    capturePhotoBtn.style.display = 'none';

    stopCameraStream();
    showToast('사진이 촬영되었습니다! 아래 [지금 인식하기]를 눌러보세요.');
  }

  function retakePhoto() {
    capturedPhotoSource = null;
    if (currentTab === 'camera') {
      uploadedImageSource = null;
    }
    cameraCapturedContainer.style.display = 'none';
    cameraStreamContainer.style.display = 'flex';
    capturePhotoBtn.style.display = 'inline-flex';
    startCameraStream();
  }

  function switchCamera() {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    startCameraStream();
    showToast(`${currentFacingMode === 'environment' ? '후면' : '전면'} 카메라로 전환되었습니다.`);
  }

  function handleCameraFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('이미지 파일만 촬영/업로드할 수 있습니다.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      capturedPhotoSource = dataUrl;
      uploadedImageSource = dataUrl;
      capturedImagePreview.src = dataUrl;
      cameraStreamContainer.style.display = 'none';
      cameraCapturedContainer.style.display = 'flex';
      capturePhotoBtn.style.display = 'none';
      stopCameraStream();
      showToast('사진 촬영이 완료되었습니다! 아래 [지금 인식하기]를 눌러보세요.');
    };
    reader.readAsDataURL(file);
  }

  capturePhotoBtn?.addEventListener('click', capturePhoto);
  retakePhotoBtn?.addEventListener('click', retakePhoto);
  switchCameraBtn?.addEventListener('click', switchCamera);
  nativeCameraBtn?.addEventListener('click', () => cameraFileInput?.click());
  cameraFileInput?.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleCameraFile(e.target.files[0]);
    }
  });

  // ==========================================
  // 드로잉 도구 이벤트 (Undo, Redo, 지우개, 비우기, 화면 이동)
  // ==========================================
  strokeWidthSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    strokeWidthVal.textContent = `${val}px`;
    drawingCanvas.setStrokeWidth(val);
  });

  undoBtn?.addEventListener('click', () => {
    const ok = drawingCanvas.undo();
    if (ok) {
      showToast('이전 획을 취소했습니다.');
    } else {
      showToast('더 이상 되돌릴 획이 없습니다.');
    }
  });

  redoBtn?.addEventListener('click', () => {
    const ok = drawingCanvas.redo();
    if (ok) {
      showToast('취소한 획을 다시 실행했습니다.');
    } else {
      showToast('다시 실행할 획이 없습니다.');
    }
  });

  clearBtn?.addEventListener('click', () => {
    drawingCanvas.clearCanvas(true);
    showToast('캔버스를 모두 비웠습니다.');
  });

  eraserBtn?.addEventListener('click', () => {
    isEraserActive = !isEraserActive;
    drawingCanvas.toggleEraser(isEraserActive);
    if (isEraserActive) {
      eraserBtn.classList.add('active');
      eraserBtn.textContent = '펜 쓰기';
    } else {
      eraserBtn.classList.remove('active');
      eraserBtn.textContent = '지우개';
    }
  });

  // ==========================================
  // 이미지 파일 업로드 로직
  // ==========================================
  uploadZone.addEventListener('click', () => fileInput.click());

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0]);
    }
  });

  function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('이미지 파일(PNG, JPG 등)만 업로드할 수 있습니다.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedImageSource = e.target.result;
      imagePreview.src = uploadedImageSource;
      uploadZone.style.display = 'none';
      uploadPreviewWrapper.style.display = 'block';
      showToast('이미지가 업로드되었습니다.');
    };
    reader.readAsDataURL(file);
  }

  removeImageBtn.addEventListener('click', () => {
    uploadedImageSource = null;
    fileInput.value = '';
    imagePreview.src = '';
    uploadPreviewWrapper.style.display = 'none';
    uploadZone.style.display = 'flex';
  });

  // ==========================================
  // 최소 글자 크기 경고 모달 이벤트
  // ==========================================
  modalRewriteBtn?.addEventListener('click', () => {
    sizeWarningModal.style.display = 'none';
    showToast('💡 점선 상자 크기보다 큼직하게 이어서 작성해 주세요.');
  });

  modalProceedBtn?.addEventListener('click', () => {
    sizeWarningModal.style.display = 'none';
    executeRecognition(true); // 강제 진행
  });

  sizeWarningModal?.addEventListener('click', (e) => {
    if (e.target === sizeWarningModal) {
      sizeWarningModal.style.display = 'none';
    }
  });

  // ==========================================
  // 인식 실행 (Main Action)
  // ==========================================
  recognizeBtn.addEventListener('click', () => {
    executeRecognition(false);
  });

  async function executeRecognition(forceProceed = false) {
    let sourceToRecognize = null;
    let canvasForAnalysis = null;

    if (currentTab === 'draw') {
      const sizeInfo = drawingCanvas.getStrokeSizeInfo();

      if (!sizeInfo.hasStroke) {
        showToast('캔버스에 글씨나 숫자를 먼저 적어주세요!');
        return;
      }

      // [1개 문자 모드일 때만] 최소 권장 크기(95px) 미달 검사
      if (currentScope === 'single' && !forceProceed && sizeInfo.isTooSmall) {
        sizeWarningModal.style.display = 'flex';
        return;
      }

      canvasForAnalysis = drawingCanvas.getPreprocessedCanvas(currentScope === 'sentence');
      sourceToRecognize = canvasForAnalysis.toDataURL('image/png');
    } else if (currentTab === 'camera') {
      if (!capturedPhotoSource) {
        showToast('손글씨 사진을 먼저 촬영해 주세요!');
        return;
      }
      sourceToRecognize = capturedPhotoSource;

      // 촬영된 사진 분석용 캔버스 생성
      try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = capturedImagePreview.naturalWidth || 640;
        tempCanvas.height = capturedImagePreview.naturalHeight || 480;
        const tctx = tempCanvas.getContext('2d');
        tctx.drawImage(capturedImagePreview, 0, 0);
        canvasForAnalysis = tempCanvas;
      } catch (e) {
        console.warn('촬영 사진 캔버스 변환 생략:', e);
      }
    } else {
      if (!uploadedImageSource) {
        showToast('인식할 손글씨 이미지를 먼저 올려주세요!');
        return;
      }
      sourceToRecognize = uploadedImageSource;

      // 업로드 이미지 분석용 캔버스 생성
      try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = imagePreview.naturalWidth || 300;
        tempCanvas.height = imagePreview.naturalHeight || 300;
        const tctx = tempCanvas.getContext('2d');
        tctx.drawImage(imagePreview, 0, 0);
        canvasForAnalysis = tempCanvas;
      } catch (e) {
        console.warn('업로드 이미지 캔버스 변환 생략:', e);
      }
    }

    // 로딩 UI 시작
    setRecognizingState(true);

    try {
      let result = null;

      // 1. Google AI 엔진 선택 시: 캔버스 획 궤적(strokes) 기반 99% 초고정밀 인식
      if (currentEngine === 'google' && currentTab === 'draw') {
        const strokes = drawingCanvas.getStrokes();
        if (strokes && strokes.length > 0) {
          try {
            result = await recognizer.recognizeWithGoogle(
              strokes,
              drawingCanvas.canvas.width,
              drawingCanvas.canvas.height,
              currentMode
            );
          } catch (gErr) {
            console.warn('Google 손글씨 API 오류, 로컬 AI로 자동 폴백:', gErr);
          }
        }
      }

      // 2. Google AI 결과가 없거나(업로드 이미지 등) 로컬 AI 엔진 선택 시: 브라우저 앙상블 실행
      if (!result) {
        result = await recognizer.recognize(sourceToRecognize, canvasForAnalysis);
        result.engine = 'local';
      }

      displayResult(result);
      addToHistory(result.text, result.confidence);
      const engineTag = result.engine === 'google' ? 'Google AI' : '로컬 AI';
      showToast(`[${engineTag}] 인식 결과: [ ${result.text} ]`);
    } catch (err) {
      console.error(err);
      showToast('인식 중 오류가 발생했습니다. 다시 시도해 주세요.');
    } finally {
      setRecognizingState(false);
    }
  }

  function setRecognizingState(isLoading) {
    recognizeBtn.disabled = isLoading;
    if (isLoading) {
      recognizeBtnText.textContent = '인공지능 정밀 분석 중...';
      recognizeBtnSpinner.style.display = 'inline-block';
      updateStatusBadge(true);
    } else {
      recognizeBtnText.textContent = '지금 인식하기';
      recognizeBtnSpinner.style.display = 'none';
      updateStatusBadge(false);
    }
  }

  function displayResult(result) {
    const isPassed = result.confidence >= currentThreshold;
    const isWord = result.isSentence || (result.text && result.text.length > 1);

    if (isWord) {
      resultCharacter.classList.add('word-mode');
    } else {
      resultCharacter.classList.remove('word-mode');
    }

    resultCharacter.innerHTML = `<span>${result.text}</span>`;
    resultCharacter.style.transform = 'scale(1.15)';
    setTimeout(() => {
      resultCharacter.style.transform = 'scale(1)';
    }, 150);

    confidencePercent.textContent = `${result.confidence}%`;
    confidenceBarFill.style.width = `${result.confidence}%`;

    // 기준치(Threshold) 충족 여부에 따른 UI 피드백
    const engineName = result.engine === 'google' ? 'Google AI' : '로컬 AI';
    if (isPassed) {
      resultCharacter.classList.remove('dimmed');
      if (thresholdAlertBox) thresholdAlertBox.style.display = 'none';
      if (statusBadge) {
        statusBadge.textContent = `${engineName} 인식 성공`;
        statusBadge.className = 'status-badge';
      }
    } else {
      resultCharacter.classList.add('dimmed');
      if (thresholdAlertBox) {
        thresholdAlertBox.style.display = 'flex';
        thresholdAlertText.textContent = `확신도(${result.confidence}%)가 설정 기준치(${currentThreshold}%)에 미달했습니다. 더 또렷하게 쓰거나 기준치를 낮춰보세요.`;
      }
      if (statusBadge) {
        statusBadge.textContent = '기준치 미달';
        statusBadge.className = 'status-badge processing';
      }
    }

    candidateList.innerHTML = '';
    if (result.candidates && result.candidates.length > 0) {
      result.candidates.forEach((cand) => {
        const item = document.createElement('div');
        item.className = 'candidate-item';
        item.innerHTML = `
          <div class="candidate-char">${cand.char}</div>
          <div class="candidate-prob">${cand.prob}</div>
        `;
        candidateList.appendChild(item);
      });
    }
  }

  // ==========================================
  // 클립보드 복사
  // ==========================================
  copyBtn.addEventListener('click', () => {
    const text = resultCharacter.innerText.trim();
    if (!text || text === '?' || text.includes('글씨를 쓰고')) {
      showToast('복사할 인식 결과가 아직 없습니다.');
      return;
    }

    navigator.clipboard.writeText(text).then(() => {
      showToast(`'${text}' 클립보드에 복사 완료!`);
    }).catch(() => {
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      showToast(`'${text}' 복사되었습니다.`);
    });
  });

  // ==========================================
  // 히스토리 관리
  // ==========================================
  function addToHistory(text, confidence) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    recognitionHistory.unshift({
      text,
      confidence,
      time: timeStr,
      id: Date.now()
    });

    if (recognitionHistory.length > 15) {
      recognitionHistory.pop();
    }

    localStorage.setItem('handwriting_history', JSON.stringify(recognitionHistory));
    renderHistory();
  }

  function renderHistory() {
    if (recognitionHistory.length === 0) {
      historyList.innerHTML = '<div class="history-empty">아직 인식한 기록이 없습니다.</div>';
      return;
    }

    historyList.innerHTML = '';
    recognitionHistory.forEach((item) => {
      const el = document.createElement('div');
      el.className = 'history-item';
      el.innerHTML = `
        <div style="display:flex; align-items:center; gap: 0.8rem;">
          <span class="history-char">${item.text}</span>
          <span class="history-meta">${item.time} (${item.confidence}%)</span>
        </div>
        <button class="history-copy" title="복사" data-copy="${item.text}">복사</button>
      `;
      historyList.appendChild(el);
    });

    historyList.querySelectorAll('.history-copy').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const textToCopy = e.currentTarget.getAttribute('data-copy');
        navigator.clipboard.writeText(textToCopy);
        showToast(`'${textToCopy}' 복사 완료!`);
      });
    });
  }

  clearHistoryBtn.addEventListener('click', () => {
    recognitionHistory = [];
    localStorage.removeItem('handwriting_history');
    renderHistory();
    showToast('기록이 모두 삭제되었습니다.');
  });

  // ==========================================
  // 토스트 메시지
  // ==========================================
  let toastTimer = null;
  function showToast(msg) {
    toastMessage.textContent = msg;
    toastNotice.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastNotice.classList.remove('show');
    }, 2400);
  }

  // ==========================================
  // 다크/라이트 테마 초기화 및 전환 (기본: 라이트 모드)
  // ==========================================
  const savedTheme = localStorage.getItem('app_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  themeToggleBtn.textContent = savedTheme === 'light' ? '다크 모드' : '라이트 모드';
  themeToggleBtn.title = savedTheme === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환';

  themeToggleBtn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('app_theme', next);
    themeToggleBtn.textContent = next === 'light' ? '다크 모드' : '라이트 모드';
    themeToggleBtn.title = next === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환';
    showToast(`${next === 'light' ? '라이트' : '다크'} 모드로 변경되었습니다.`);
  });

  // ==========================================
  // 접속QR코드 보드 모달 제어
  // ==========================================
  function openQrModal() {
    if (qrModal) qrModal.style.display = 'flex';
  }

  function closeQrModal() {
    if (qrModal) qrModal.style.display = 'none';
  }

  qrModalBtn?.addEventListener('click', openQrModal);
  qrModalCloseBtn?.addEventListener('click', closeQrModal);
  qrModalCloseXBtn?.addEventListener('click', closeQrModal);

  qrModal?.addEventListener('click', (e) => {
    if (e.target === qrModal) {
      closeQrModal();
    }
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && qrModal && qrModal.style.display === 'flex') {
      closeQrModal();
    }
  });
});
