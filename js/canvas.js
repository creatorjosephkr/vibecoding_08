/**
 * Canvas Drawing & Preprocessing Module (Enhanced Precision)
 * 마우스 포인트와 드로잉 위치가 100% 정확하게 일치하도록 정밀 좌표 변환 적용
 */

class DrawingCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    
    // 드로잉 상태
    this.isDrawing = false;
    this.strokeColor = '#000000';
    this.strokeWidth = 20; // 기본 펜 굵기 (CSS 픽셀 기준)
    this.isEraser = false;
    this.dpr = window.devicePixelRatio || 1;
    
    // 점 보간을 위한 좌표 배열
    this.points = [];
    
    // 실행 취소(Undo) 히스토리 스택
    this.history = [];
    this.maxHistory = 20;

    // 초기화
    this.initCanvasSize();
    this.bindEvents();
    this.saveState();
  }

  initCanvasSize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width || 440;
    const height = rect.height || 340;
    this.dpr = window.devicePixelRatio || 1;

    // 캔버스 버퍼 해상도를 실제 디스플레이 비율(Retina 등)에 정확히 맞춤
    this.canvas.width = Math.round(width * this.dpr);
    this.canvas.height = Math.round(height * this.dpr);

    this.clearCanvas(false);
  }

  bindEvents() {
    // 윈도우 리사이즈 및 모바일 화면 회전 시 비율 재계산 (기존 그림 보존)
    let resizeTimer = null;
    const onResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        this.handleResize();
      }, 150);
    };
    window.addEventListener('resize', onResize);
    window.addEventListener('orientationchange', onResize);

    // 마우스 이벤트
    this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
    this.canvas.addEventListener('mousemove', (e) => this.draw(e));
    window.addEventListener('mouseup', () => this.stopDrawing());
    this.canvas.addEventListener('mouseleave', () => this.stopDrawing());

    // 모바일 터치 이벤트 (화면 스크롤 간섭 방지)
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      this.startDrawing(touch);
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      this.draw(touch);
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.stopDrawing();
    });

    this.canvas.addEventListener('touchcancel', () => {
      this.stopDrawing();
    });
  }

  handleResize() {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    // 임시 복사
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = this.canvas.width;
    tempCanvas.height = this.canvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(this.canvas, 0, 0);

    const oldW = this.canvas.width;
    const oldH = this.canvas.height;

    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);

    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.drawImage(tempCanvas, 0, 0, oldW, oldH, 0, 0, this.canvas.width, this.canvas.height);

    this.saveState();
  }

  /**
   * 마우스/터치 위치를 캔버스 내부 비트맵 해상도 좌표로 100% 정밀 변환
   */
  getPointerPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  startDrawing(e) {
    this.isDrawing = true;
    const pos = this.getPointerPos(e);
    this.points = [pos];

    const currentLineWidth = (this.isEraser ? this.strokeWidth * 2.2 : this.strokeWidth) * this.dpr;

    this.ctx.beginPath();
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.lineWidth = currentLineWidth;
    this.ctx.strokeStyle = this.isEraser ? '#ffffff' : this.strokeColor;
    
    // 시작점에 둥근 점 찍기
    this.ctx.arc(pos.x, pos.y, currentLineWidth / 2, 0, Math.PI * 2);
    this.ctx.fillStyle = this.ctx.strokeStyle;
    this.ctx.fill();
    this.ctx.beginPath();
  }

  draw(e) {
    if (!this.isDrawing) return;
    const pos = this.getPointerPos(e);
    this.points.push(pos);

    const currentLineWidth = (this.isEraser ? this.strokeWidth * 2.2 : this.strokeWidth) * this.dpr;
    this.ctx.lineWidth = currentLineWidth;
    this.ctx.strokeStyle = this.isEraser ? '#ffffff' : this.strokeColor;

    if (this.points.length < 3) {
      const b = this.points[0];
      this.ctx.beginPath();
      this.ctx.arc(b.x, b.y, currentLineWidth / 2, 0, Math.PI * 2);
      this.ctx.fill();
      return;
    }

    // 이차 베지어 곡선으로 획을 부드럽게 연결
    this.ctx.beginPath();
    this.ctx.moveTo(this.points[0].x, this.points[0].y);

    for (let i = 1; i < this.points.length - 2; i++) {
      const c = (this.points[i].x + this.points[i + 1].x) / 2;
      const d = (this.points[i].y + this.points[i + 1].y) / 2;
      this.ctx.quadraticCurveTo(this.points[i].x, this.points[i].y, c, d);
    }

    const lastIdx = this.points.length - 2;
    this.ctx.quadraticCurveTo(
      this.points[lastIdx].x,
      this.points[lastIdx].y,
      this.points[lastIdx + 1].x,
      this.points[lastIdx + 1].y
    );

    this.ctx.stroke();
  }

  stopDrawing() {
    if (this.isDrawing) {
      this.isDrawing = false;
      this.points = [];
      this.saveState();
    }
  }

  saveState() {
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    if (this.history.length >= this.maxHistory) {
      this.history.shift();
    }
    this.history.push(imgData);
  }

  undo() {
    if (this.history.length > 1) {
      this.history.pop();
      const prevState = this.history[this.history.length - 1];
      this.ctx.putImageData(prevState, 0, 0);
      return true;
    } else if (this.history.length === 1) {
      this.clearCanvas(false);
      return true;
    }
    return false;
  }

  clearCanvas(save = true) {
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    if (save) {
      this.saveState();
    }
  }

  setStrokeWidth(val) {
    this.strokeWidth = parseInt(val, 10) || 20;
  }

  toggleEraser(active) {
    this.isEraser = active;
  }

  isEmpty() {
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] < 220 || data[i + 1] < 220 || data[i + 2] < 220) {
        return false;
      }
    }
    return true;
  }

  /**
   * 작성된 글자의 실제 크기(바운딩 박스) 및 최소 권장 크기 미달 여부 검사
   */
  getStrokeSizeInfo() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const imgData = this.ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let minX = width, minY = height, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (brightness < 210) {
          found = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (!found) {
      return { hasStroke: false, widthCss: 0, heightCss: 0, isTooSmall: false };
    }

    // CSS 픽셀 단위로 환산
    const strokeWidthCss = (maxX - minX) / this.dpr;
    const strokeHeightCss = (maxY - minY) / this.dpr;
    const maxDimCss = Math.max(strokeWidthCss, strokeHeightCss);

    // 가이드 점선 상자 기준, 최소 권장 크기는 약 95px 이상
    const minRequiredPx = 95;
    const isTooSmall = maxDimCss < minRequiredPx;

    return {
      hasStroke: true,
      widthCss: Math.round(strokeWidthCss),
      heightCss: Math.round(strokeHeightCss),
      maxDimCss: Math.round(maxDimCss),
      isTooSmall: isTooSmall,
      minRequiredPx: minRequiredPx
    };
  }

  /**
   * 고화질 정규화 전처리 캔버스 객체 반환
   * @param {boolean} isSentence 단어 또는 문장 모드 여부
   */
  getPreprocessedCanvas(isSentence = false) {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const imgData = this.ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let minX = width, minY = height, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (brightness < 210) {
          found = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (!found) {
      const emptyCanvas = document.createElement('canvas');
      emptyCanvas.width = 300;
      emptyCanvas.height = 300;
      const ectx = emptyCanvas.getContext('2d');
      ectx.fillStyle = '#ffffff';
      ectx.fillRect(0, 0, 300, 300);
      return emptyCanvas;
    }

    const boxW = maxX - minX;
    const boxH = maxY - minY;

    // [단어 또는 문장 모드] 가로 비율을 그대로 보존하면서 상하좌우 여백을 준 고해상도 캔버스 생성
    if (isSentence) {
      // Tesseract가 가장자리를 텍스트로 오인하지 않고, 잘리지 않도록 넉넉한 Quiet Zone 여백 제공
      const pad = 65;
      const targetH = Math.max(Math.min(boxH, 180), 130);
      const scale = targetH / Math.max(boxH, 1);
      const outW = Math.round(boxW * scale) + pad * 2;
      const outH = Math.round(boxH * scale) + pad * 2;

      const sentCanvas = document.createElement('canvas');
      sentCanvas.width = Math.max(outW, 260);
      sentCanvas.height = Math.max(outH, 140);
      const sctx = sentCanvas.getContext('2d');

      sctx.fillStyle = '#ffffff';
      sctx.fillRect(0, 0, sentCanvas.width, sentCanvas.height);

      const drawX = (sentCanvas.width - (boxW * scale)) / 2;
      const drawY = (sentCanvas.height - (boxH * scale)) / 2;

      sctx.drawImage(
        this.canvas,
        minX, minY, boxW, boxH,
        drawX, drawY, boxW * scale, boxH * scale
      );

      // 부드러운 안티에일리어싱 명암 보정 (계단 현상 방지 & 신경망 특징점 보존)
      const sImgData = sctx.getImageData(0, 0, sentCanvas.width, sentCanvas.height);
      const sData = sImgData.data;
      for (let i = 0; i < sData.length; i += 4) {
        const avg = (sData[i] + sData[i + 1] + sData[i + 2]) / 3;
        if (avg >= 225) {
          sData[i] = 255;
          sData[i + 1] = 255;
          sData[i + 2] = 255;
        } else if (avg <= 110) {
          sData[i] = 0;
          sData[i + 1] = 0;
          sData[i + 2] = 0;
        } else {
          // 중간 계조 부드러운 S-커브 대비
          const norm = (avg - 110) / 115;
          const val = Math.round(norm * norm * 255);
          sData[i] = val;
          sData[i + 1] = val;
          sData[i + 2] = val;
        }
        sData[i + 3] = 255;
      }
      sctx.putImageData(sImgData, 0, 0);
      return sentCanvas;
    }

    // [1개 문자 모드] 300x300 정사각형 중앙 정렬 (MNIST CNN 및 단일 문자 최적화)
    const procCanvas = document.createElement('canvas');
    procCanvas.width = 300;
    procCanvas.height = 300;
    const pctx = procCanvas.getContext('2d');

    pctx.fillStyle = '#ffffff';
    pctx.fillRect(0, 0, 300, 300);

    const maxDim = Math.max(boxW, boxH);
    const targetSize = 210;
    const scale = targetSize / maxDim;
    const drawW = boxW * scale;
    const drawH = boxH * scale;
    const drawX = (300 - drawW) / 2;
    const drawY = (300 - drawH) / 2;

    pctx.drawImage(
      this.canvas,
      minX, minY, boxW, boxH,
      drawX, drawY, drawW, drawH
    );

    // 명암 대비 이진화
    const pData = pctx.getImageData(0, 0, 300, 300);
    const px = pData.data;
    for (let i = 0; i < px.length; i += 4) {
      const avg = (px[i] + px[i + 1] + px[i + 2]) / 3;
      if (avg < 205) {
        px[i] = 0;
        px[i + 1] = 0;
        px[i + 2] = 0;
      } else {
        px[i] = 255;
        px[i + 1] = 255;
        px[i + 2] = 255;
      }
      px[i + 3] = 255;
    }
    pctx.putImageData(pData, 0, 0);

    return procCanvas;
  }

  getPreprocessedDataURL() {
    const procCanvas = this.getPreprocessedCanvas();
    return procCanvas.toDataURL('image/png');
  }

  /**
   * 샘플 글씨 그리기 (1글자 또는 단어/문장 적응형 크기)
   */
  drawSample(text) {
    this.clearCanvas(false);
    this.ctx.save();
    this.ctx.fillStyle = '#000000';

    const len = text.length;
    let baseFontSize = 140;
    if (len === 2) baseFontSize = 110;
    else if (len >= 3 && len <= 5) baseFontSize = 85;
    else if (len > 5) baseFontSize = 55;

    this.ctx.font = `bold ${Math.round(baseFontSize * this.dpr)}px "Comic Sans MS", "Caveat", "Segoe Script", sans-serif`;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;

    this.ctx.fillText(text, cx, cy);
    this.ctx.restore();
    this.saveState();
  }
}

window.DrawingCanvas = DrawingCanvas;
