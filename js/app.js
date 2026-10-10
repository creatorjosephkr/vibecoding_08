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
  const tabUpload = document.getElementById('tabUpload');
  const canvasWrapper = document.getElementById('canvasWrapper');
  const canvasToolbar = document.getElementById('canvasToolbar');
  const canvasGuideHint = document.getElementById('canvasGuideHint');

  // Scope & Mode Elements
  const scopeSingleBtn = document.getElementById('scopeSingleBtn');
  const scopeSentenceBtn = document.getElementById('scopeSentenceBtn');
  const minSizeGuideBox = document.getElementById('minSizeGuideBox');
  const sentenceGuideLines = document.getElementById('sentenceGuideLines');

  const modeDigitsBtn = document.getElementById('modeDigitsBtn');
  const modeLettersBtn = document.getElementById('modeLettersBtn');
  const modeAllBtn = document.getElementById('modeAllBtn');
  const modeButtons = [modeDigitsBtn, modeLettersBtn, modeAllBtn];

  const strokeWidthSlider = document.getElementById('strokeWidth');
  const strokeWidthVal = document.getElementById('strokeWidthVal');
  const undoBtn = document.getElementById('undoBtn');
  const clearBtn = document.getElementById('clearBtn');
  const eraserBtn = document.getElementById('eraserBtn');

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

  const themeToggleBtn = document.getElementById('themeToggleBtn');

  // Instances
  const drawingCanvas = new DrawingCanvas('drawingCanvas');
  const recognizer = new HandwritingRecognizer();

  // App State
  let currentTab = 'draw'; // 'draw' | 'upload'
  let currentScope = 'single'; // 'single': 1개 문자 | 'sentence': 단어/문장
  let currentMode = 'digits'; // 'digits' | 'letters' | 'all'
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

    const scopeLabel = currentScope === 'single' ? '1개 문자' : '단어/문장';
    const modeLabel = currentMode === 'digits' ? '숫자' : currentMode === 'letters' ? '영문' : '통합';
    statusBadge.textContent = `준비 완료 (${scopeLabel} · ${modeLabel})`;
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
        canvasGuideHint.innerHTML = '<span>📐 1개 문자 모드:</span> 캔버스 중앙 점선 상자보다 <strong>큼직하게 1글자</strong> 작성해 주세요';
      }
      // 1개 문자 최적 펜 굵기 복원
      if (strokeWidthSlider && parseInt(strokeWidthSlider.value, 10) < 18) {
        strokeWidthSlider.value = 22;
        if (strokeWidthVal) strokeWidthVal.textContent = '22px';
        drawingCanvas.setStrokeWidth(22);
      }
      showToast('🔤 [1개 문자 모드] 한 글자씩 딥러닝/OCR로 정밀 판독합니다.');
    } else {
      if (minSizeGuideBox) minSizeGuideBox.style.display = 'none';
      if (sentenceGuideLines) sentenceGuideLines.style.display = 'flex';
      if (canvasGuideHint) {
        canvasGuideHint.innerHTML = '<span>📝 단어/문장 모드:</span> 가로 방향으로 <strong>단어나 문장</strong>을 자유롭게 이어서 작성해 주세요';
      }
      // 연속 단어 필기 시 글자 뭉침 방지를 위해 알맞은 펜 굵기(14px)로 자동 최적화
      if (strokeWidthSlider && parseInt(strokeWidthSlider.value, 10) > 16) {
        strokeWidthSlider.value = 14;
        if (strokeWidthVal) strokeWidthVal.textContent = '14px';
        drawingCanvas.setStrokeWidth(14);
      }
      showToast('📝 [단어/문장 모드] 여러 글자나 문장을 가로로 이어서 인식합니다.');
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
      showToast(`🎯 인식 기준치가 ${val}%로 설정되었습니다.`);
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
      showToast('🔢 [숫자] 모드: 0~9 손글씨에 최적화되었습니다.');
    } else if (mode === 'letters') {
      showToast('🔤 [영문] 모드: A~Z 알파벳에 최적화되었습니다.');
    } else {
      showToast('🌐 [통합] 모드: 숫자와 영문을 함께 판별합니다.');
    }

    updateStatusBadge();
  }

  modeDigitsBtn?.addEventListener('click', () => setRecognitionMode('digits'));
  modeLettersBtn?.addEventListener('click', () => setRecognitionMode('letters'));
  modeAllBtn?.addEventListener('click', () => setRecognitionMode('all'));

  // ==========================================
  // 탭 전환 로직
  // ==========================================
  function switchTab(tab) {
    currentTab = tab;
    if (tab === 'draw') {
      tabDraw.classList.add('active');
      tabUpload.classList.remove('active');
      canvasWrapper.style.display = 'flex';
      canvasToolbar.style.display = 'flex';
      if (canvasGuideHint) canvasGuideHint.style.display = 'flex';
      uploadZone.style.display = 'none';
      uploadPreviewWrapper.style.display = 'none';
      setTimeout(() => drawingCanvas.handleResize(), 30);
    } else {
      tabUpload.classList.add('active');
      tabDraw.classList.remove('active');
      canvasWrapper.style.display = 'none';
      canvasToolbar.style.display = 'none';
      if (canvasGuideHint) canvasGuideHint.style.display = 'none';

      if (uploadedImageSource) {
        uploadZone.style.display = 'none';
        uploadPreviewWrapper.style.display = 'block';
      } else {
        uploadZone.style.display = 'flex';
        uploadPreviewWrapper.style.display = 'none';
      }
    }
  }

  tabDraw.addEventListener('click', () => switchTab('draw'));
  tabUpload.addEventListener('click', () => switchTab('upload'));

  // ==========================================
  // 드로잉 도구 이벤트
  // ==========================================
  strokeWidthSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    strokeWidthVal.textContent = `${val}px`;
    drawingCanvas.setStrokeWidth(val);
  });

  undoBtn.addEventListener('click', () => {
    drawingCanvas.undo();
  });

  clearBtn.addEventListener('click', () => {
    drawingCanvas.clearCanvas(true);
  });

  eraserBtn.addEventListener('click', () => {
    isEraserActive = !isEraserActive;
    drawingCanvas.toggleEraser(isEraserActive);
    if (isEraserActive) {
      eraserBtn.classList.add('active');
      eraserBtn.innerHTML = '✏️ 펜 모드로 전환';
    } else {
      eraserBtn.classList.remove('active');
      eraserBtn.innerHTML = '🧹 지우개';
    }
  });

  // 샘플 글씨 칩 클릭
  document.querySelectorAll('.chip-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const char = btn.getAttribute('data-char');
      const reqScope = btn.getAttribute('data-scope');
      switchTab('draw');

      // 단어/문장 칩 여부에 따라 스코프 자동 전환
      if (reqScope === 'sentence') {
        setRecognitionScope('sentence');
        if (/^[0-9\s]+$/.test(char)) {
          setRecognitionMode('digits');
        } else if (/^[A-Za-z\s]+$/.test(char)) {
          setRecognitionMode('letters');
        } else {
          setRecognitionMode('all');
        }
      } else {
        setRecognitionScope('single');
        if (currentMode !== 'all') {
          if (/^[0-9]$/.test(char) && currentMode !== 'digits') {
            setRecognitionMode('digits');
          } else if (/^[A-Za-z]$/.test(char) && currentMode !== 'letters') {
            setRecognitionMode('letters');
          }
        }
      }

      drawingCanvas.drawSample(char);
      showToast(`'${char}' 예시 글씨를 캔버스에 그렸습니다.`);
    });
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
      const result = await recognizer.recognize(sourceToRecognize, canvasForAnalysis);
      displayResult(result);
      addToHistory(result.text, result.confidence);
      showToast(`인식 결과: [ ${result.text} ]`);
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
      recognizeBtnText.textContent = '⚡ 지금 인식하기';
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
    if (isPassed) {
      resultCharacter.classList.remove('dimmed');
      if (thresholdAlertBox) thresholdAlertBox.style.display = 'none';
      if (statusBadge) {
        statusBadge.textContent = '인식 성공';
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
        <button class="history-copy" title="복사" data-copy="${item.text}">📋</button>
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
  themeToggleBtn.innerHTML = savedTheme === 'light' ? '🌙' : '☀️';
  themeToggleBtn.title = savedTheme === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환';

  themeToggleBtn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('app_theme', next);
    themeToggleBtn.innerHTML = next === 'light' ? '🌙' : '☀️';
    themeToggleBtn.title = next === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환';
    showToast(`${next === 'light' ? '라이트' : '다크'} 모드로 변경되었습니다.`);
  });
});
