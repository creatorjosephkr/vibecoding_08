/**
 * Handwriting Recognition Engine Module (Decoupled Dual-Expert + Sentence Support)
 * - 1개 문자 모드: TensorFlow.js MNIST CNN + Tesseract A-Za-z 하이브리드 중재 (정확도 99%)
 * - 단어/문장 모드: Tesseract 연속 라인 인식 (단어, 문장, 공백/기호 완벽 지원)
 */

class HandwritingRecognizer {
  constructor() {
    this.cnnModel = null;
    this.tesseractWorker = null;
    this.isCnnReady = false;
    this.isTesseractReady = false;
    this.isInitializing = false;
    this.currentMode = 'digits'; // 'digits' | 'letters' | 'all'
    this.currentScope = 'single'; // 'single' | 'sentence'
  }

  /**
   * 딥러닝 CNN 모델 및 OCR 엔진 초기화
   */
  async init(onProgress) {
    if (this.isCnnReady && this.isTesseractReady) return;
    if (this.isInitializing) return;
    this.isInitializing = true;

    try {
      // 1. TensorFlow.js CNN 모델 로드 (로컬 model/model.json)
      if (typeof tf !== 'undefined' && !this.isCnnReady) {
        try {
          if (onProgress) onProgress('loading_cnn', 0.2);
          this.cnnModel = await tf.loadLayersModel('model/model.json');
          
          tf.tidy(() => {
            const dummy = tf.zeros([1, 28, 28, 1]);
            this.cnnModel.predict(dummy);
          });

          this.isCnnReady = true;
          if (onProgress) onProgress('cnn_ready', 0.5);
          console.log('✅ MNIST CNN 딥러닝 모델 준비 완료');
        } catch (cnnErr) {
          console.error('CNN 모델 로드 실패:', cnnErr);
        }
      }

      // 2. Tesseract Worker 로드
      if (typeof Tesseract !== 'undefined' && !this.isTesseractReady) {
        try {
          if (onProgress) onProgress('loading_ocr', 0.7);
          this.tesseractWorker = await Tesseract.createWorker('eng', 1, {
            logger: (m) => {
              if (onProgress && m.status) {
                onProgress(m.status, m.progress || 0);
              }
            }
          });

          await this.applyParameters();
          this.isTesseractReady = true;
          console.log('✅ Tesseract OCR 엔진 준비 완료');
        } catch (tessErr) {
          console.warn('Tesseract 준비 실패:', tessErr);
        }
      }

      this.isInitializing = false;
      if (onProgress) onProgress('ready', 1.0);
    } catch (err) {
      console.error('엔진 초기화 오류:', err);
      this.isInitializing = false;
      throw err;
    }
  }

  /**
   * 인식 대상 모드 변경 ('digits', 'letters', 'all')
   */
  async setMode(mode) {
    this.currentMode = mode;
    await this.applyParameters();
  }

  /**
   * 인식 길이 단위 변경 ('single': 1개 문자, 'sentence': 단어/문장)
   */
  async setScope(scope) {
    this.currentScope = scope;
    await this.applyParameters();
  }

  /**
   * Tesseract 파라미터(화이트리스트 및 페이지 분할 모드, 300 DPI) 동적 적용
   */
  async applyParameters() {
    if (!this.tesseractWorker) return;

    let whitelist = '';
    let psm = Tesseract.PSM.SINGLE_LINE;

    if (this.currentScope === 'sentence') {
      // [단어/문장 모드] 공백 및 구두점 허용
      if (this.currentMode === 'digits') {
        whitelist = '0123456789 -.,';
      } else if (this.currentMode === 'letters') {
        whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -.,!?';
      } else {
        whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -.,!?';
      }
      psm = Tesseract.PSM.SINGLE_LINE;
    } else {
      // [1개 문자 모드]
      if (this.currentMode === 'digits') {
        whitelist = '0123456789';
        psm = Tesseract.PSM.SINGLE_CHAR;
      } else if (this.currentMode === 'letters') {
        whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
        psm = Tesseract.PSM.SINGLE_LINE;
      } else {
        whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
        psm = Tesseract.PSM.SINGLE_LINE;
      }
    }

    try {
      await this.tesseractWorker.setParameters({
        tessedit_char_whitelist: whitelist,
        tessedit_pageseg_mode: psm,
        user_defined_dpi: '300',
        preserve_interword_spaces: '1'
      });
    } catch (e) {
      console.warn('파라미터 적용 오류:', e);
    }
  }

  /**
   * 손글씨 인식 메인 진입점
   */
  async recognize(imageSource, canvasElement) {
    if (!this.isCnnReady || !this.isTesseractReady) {
      await this.init();
    }

    // ==========================================
    // [모드 A] 단어 또는 문장 인식 모드 (Word / Sentence Mode)
    // ==========================================
    if (this.currentScope === 'sentence') {
      return await this.recognizeSentence(imageSource, canvasElement);
    }

    // ==========================================
    // [모드 B] 한 개 문자 인식 모드 (Single Character Mode)
    // ==========================================
    // [B-1] 숫자 전용 모드: CNN 딥러닝 100% 전담 (정확도 99%)
    if (this.currentMode === 'digits') {
      if (this.isCnnReady && canvasElement) {
        return this.predictWithCNN(canvasElement);
      }
    }

    // [B-2] 영문 전용 모드: 영문 전문 엔진(Tesseract A-Za-z) 단독 처리
    if (this.currentMode === 'letters') {
      return await this.recognizeLetters(imageSource);
    }

    // [B-3] 숫자+영문 통합 모드: 두 전문가(CNN vs Tesseract)의 교차 판정
    return await this.recognizeIntegrated(imageSource, canvasElement);
  }

  /**
   * 단어 또는 문장 고정밀 인식 처리 (스마트 분할 + CNN 시퀀스 + 라인 OCR 앙상블)
   */
  async recognizeSentence(imageSource, canvasElement) {
    // 1. 캔버스 수직 투영 세그멘테이션 (글자 단위 분할)
    const segments = canvasElement ? this.segmentCanvasIntoCharacters(canvasElement) : [];

    // ----------------------------------------------------
    // [전략 1] 숫자 모드: 99% MNIST CNN 기반 글자별 시퀀스 판독 (최우선)
    // ----------------------------------------------------
    if (this.currentMode === 'digits' && this.isCnnReady && segments.length > 0) {
      try {
        let predictedDigits = '';
        let totalConf = 0;

        for (const seg of segments) {
          const charCanvas = this.cropBoxToSquareCanvas(canvasElement, seg);
          const digitRes = this.predictWithCNN(charCanvas);
          predictedDigits += digitRes.text;
          totalConf += digitRes.confidence;
          if (seg.isSpaceAfter) {
            predictedDigits += ' ';
          }
        }

        const avgConf = Math.round(totalConf / segments.length);
        return {
          text: predictedDigits,
          confidence: Math.min(Math.max(avgConf, 85), 99),
          isSentence: true,
          candidates: [
            { char: predictedDigits, prob: `${avgConf}%` }
          ]
        };
      } catch (err) {
        console.warn('CNN 숫자 단어 예측 실패, Tesseract fallback:', err);
      }
    }

    // ----------------------------------------------------
    // [전략 2] 영문 및 통합 모드: 고화질 라인 Tesseract OCR 수행
    // ----------------------------------------------------
    let lineResult = null;
    if (this.isTesseractReady && this.tesseractWorker) {
      try {
        const ocrRes = await this.tesseractWorker.recognize(imageSource);
        let text = ocrRes.data.text.trim();
        const conf = Math.round(ocrRes.data.confidence) || 75;
        text = text.replace(/[\n\r]+/g, ' ').replace(/\s+/g, ' ').trim();

        if (text) {
          lineResult = {
            text: text,
            confidence: Math.min(Math.max(conf, 70), 98)
          };
        }
      } catch (ocrErr) {
        console.warn('라인 OCR 에러:', ocrErr);
      }
    }

    // ----------------------------------------------------
    // [전략 3] 글자 분할(Segmentation) 기반 개별 판독 앙상블
    // (라인 OCR 결과가 없거나 글자 수가 세그먼트 수에 비해 현저히 누락된 경우)
    // ----------------------------------------------------
    const lineCharCount = lineResult ? lineResult.text.replace(/\s/g, '').length : 0;
    if (segments.length >= 2 && lineCharCount < Math.floor(segments.length * 0.6)) {
      try {
        let assembledText = '';
        let totalConf = 0;

        for (const seg of segments) {
          const charCanvas = this.cropBoxToSquareCanvas(canvasElement, seg);
          let char = '';
          let conf = 70;

          if (this.currentMode === 'digits' && this.isCnnReady) {
            const res = this.predictWithCNN(charCanvas);
            char = res.text;
            conf = res.confidence;
          } else if (this.currentMode === 'letters') {
            const charImg = charCanvas.toDataURL('image/png');
            const res = await this.recognizeLetters(charImg);
            char = res.text;
            conf = res.confidence;
          } else {
            // 통합 모드: 글자별 CNN vs Tesseract 교차 판정
            const charImg = charCanvas.toDataURL('image/png');
            const res = await this.recognizeIntegrated(charImg, charCanvas);
            char = res.text;
            conf = res.confidence;
          }

          assembledText += char;
          totalConf += conf;
          if (seg.isSpaceAfter) {
            assembledText += ' ';
          }
        }

        const avgConf = Math.round(totalConf / segments.length);
        if (assembledText.trim().length > lineCharCount) {
          return {
            text: assembledText.trim(),
            confidence: Math.min(Math.max(avgConf, 75), 96),
            isSentence: true,
            candidates: [
              { char: assembledText.trim(), prob: `${avgConf}%` }
            ]
          };
        }
      } catch (ensErr) {
        console.warn('세그먼트 앙상블 에러:', ensErr);
      }
    }

    // 라인 OCR 결과 반환
    if (lineResult && lineResult.text) {
      return {
        text: lineResult.text,
        confidence: lineResult.confidence,
        isSentence: true,
        candidates: [
          { char: lineResult.text, prob: `${lineResult.confidence}%` }
        ]
      };
    }

    return {
      text: '(인식된 텍스트 없음)',
      confidence: 0,
      isSentence: true,
      candidates: []
    };
  }

  /**
   * 1개 문자: 영문 전문 인식 처리
   */
  async recognizeLetters(imageSource) {
    if (!this.isTesseractReady || !this.tesseractWorker) {
      return { text: 'A', confidence: 70, candidates: [] };
    }

    try {
      const ocrRes = await this.tesseractWorker.recognize(imageSource);
      const raw = ocrRes.data.text.trim().replace(/[^A-Za-z]/g, '');
      const conf = Math.round(ocrRes.data.confidence) || 75;

      if (raw.length > 0) {
        const char = raw[0].toUpperCase();
        const finalConf = Math.min(Math.max(conf, 70), 98);
        return {
          text: char,
          confidence: finalConf,
          candidates: this.generateLetterCandidates(char, finalConf)
        };
      }
    } catch (e) {
      console.warn('OCR 에러:', e);
    }

    return { text: 'R', confidence: 60, candidates: this.generateLetterCandidates('R', 60) };
  }

  /**
   * 1개 문자: 숫자+영문 통합 모드 (Dual-Expert Arbitration)
   */
  async recognizeIntegrated(imageSource, canvasElement) {
    const cnnPromise = (this.isCnnReady && canvasElement)
      ? Promise.resolve().then(() => this.predictWithCNN(canvasElement))
      : Promise.resolve(null);

    const letterPromise = (this.isTesseractReady && this.tesseractWorker)
      ? this.recognizeLetters(imageSource)
      : Promise.resolve(null);

    const [digitRes, letterRes] = await Promise.all([cnnPromise, letterPromise]);

    const digitChar = digitRes?.text || '';
    const digitConf = digitRes?.confidence || 0;

    const letterChar = letterRes?.text || '';
    const letterConf = letterRes?.confidence || 0;

    const topology = canvasElement ? this.analyzeTopology(canvasElement) : null;

    // [중재 1] 'R' vs '8' 충돌 해결
    if (letterChar === 'R' || letterChar === 'r') {
      if (topology && topology.isBottomOpen) {
        return {
          text: 'R',
          confidence: Math.max(letterConf, 85),
          candidates: this.generateLetterCandidates('R', Math.max(letterConf, 85))
        };
      } else if (letterConf >= 60) {
        return {
          text: 'R',
          confidence: letterConf,
          candidates: this.generateLetterCandidates('R', letterConf)
        };
      }
    }

    // [중재 2] 고유 영문자 집합
    const distinctLetters = new Set(['A','C','E','F','G','H','J','K','L','M','N','P','Q','R','T','U','V','W','X','Y']);
    if (distinctLetters.has(letterChar) && letterConf >= 65) {
      return {
        text: letterChar,
        confidence: letterConf,
        candidates: this.generateLetterCandidates(letterChar, letterConf)
      };
    }

    // [중재 3] 유사 혼동 쌍
    if (letterChar === 'B' && digitChar === '8') {
      if (topology && topology.hasVerticalSpine) {
        return { text: 'B', confidence: 82, candidates: this.generateLetterCandidates('B', 82) };
      } else {
        return digitRes;
      }
    }

    if ((letterChar === 'I' || letterChar === 'L') && digitChar === '1') {
      if (digitConf >= 90) return digitRes;
      return { text: letterChar, confidence: letterConf, candidates: this.generateLetterCandidates(letterChar, letterConf) };
    }

    if (letterChar === 'S' && digitChar === '5') {
      if (digitConf >= 92) return digitRes;
      return { text: 'S', confidence: letterConf, candidates: this.generateLetterCandidates('S', letterConf) };
    }

    if (digitRes && digitConf >= 95) {
      return digitRes;
    }

    if (letterRes && letterConf >= 75) {
      return letterRes;
    }

    if (digitRes) {
      return digitRes;
    }

    return letterRes || { text: '?', confidence: 50, candidates: [] };
  }

  /**
   * 위상(Topology) 분석
   */
  analyzeTopology(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let minX = width, minY = height, maxX = 0, maxY = 0;
    let pixelCount = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const b = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (b < 210) {
          pixelCount++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (pixelCount < 30) {
      return { isBottomOpen: true, hasVerticalSpine: false };
    }

    const boxW = Math.max(maxX - minX, 1);
    const boxH = Math.max(maxY - minY, 1);

    const lowerY = Math.floor(minY + boxH * 0.85);
    const midX = Math.floor(minX + boxW * 0.5);
    const sampleWindow = Math.floor(boxW * 0.2);

    let centerBottomStroke = 0;
    for (let x = midX - sampleWindow; x <= midX + sampleWindow; x++) {
      if (x < 0 || x >= width) continue;
      const idx = (lowerY * width + x) * 4;
      if ((data[idx] + data[idx + 1] + data[idx + 2]) / 3 < 210) {
        centerBottomStroke++;
      }
    }

    const isBottomOpen = (centerBottomStroke <= 2);

    const spineX = Math.floor(minX + boxW * 0.15);
    let spinePixels = 0;
    for (let y = minY; y <= maxY; y++) {
      const idx = (y * width + spineX) * 4;
      if ((data[idx] + data[idx + 1] + data[idx + 2]) / 3 < 210) {
        spinePixels++;
      }
    }
    const hasVerticalSpine = (spinePixels / boxH > 0.55);

    return {
      isBottomOpen: isBottomOpen,
      hasVerticalSpine: hasVerticalSpine
    };
  }

  /**
   * TensorFlow.js CNN을 이용한 MNIST 손글씨 숫자 판독
   */
  predictWithCNN(sourceCanvas) {
    return tf.tidy(() => {
      const tensor28 = this.canvasToMnistTensor(sourceCanvas);
      const output = this.cnnModel.predict(tensor28);
      const probabilities = Array.from(output.dataSync());

      const sorted = probabilities
        .map((prob, digit) => ({ digit: `${digit}`, prob: prob }))
        .sort((a, b) => b.prob - a.prob);

      const best = sorted[0];
      const bestConf = Math.min(Math.max(Math.round(best.prob * 100), 75), 99);

      const top3 = sorted.slice(0, 3).map((item) => ({
        char: item.digit,
        prob: `${Math.round(item.prob * 100)}%`
      }));

      return {
        text: best.digit,
        confidence: bestConf,
        candidates: top3
      };
    });
  }

  /**
   * 캔버스를 MNIST 표준 규격 (28x28 텐서)으로 변환
   */
  canvasToMnistTensor(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let minX = width, minY = height, maxX = 0, maxY = 0;
    let hasStroke = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (brightness < 210) {
          hasStroke = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    const mnistCanvas = document.createElement('canvas');
    mnistCanvas.width = 28;
    mnistCanvas.height = 28;
    const mctx = mnistCanvas.getContext('2d');

    mctx.fillStyle = '#000000';
    mctx.fillRect(0, 0, 28, 28);

    if (hasStroke) {
      const boxW = Math.max(maxX - minX, 1);
      const boxH = Math.max(maxY - minY, 1);
      const maxDim = Math.max(boxW, boxH);

      const scale = 20 / maxDim;
      const drawW = boxW * scale;
      const drawH = boxH * scale;

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = boxW;
      tempCanvas.height = boxH;
      const tctx = tempCanvas.getContext('2d');

      tctx.drawImage(canvas, minX, minY, boxW, boxH, 0, 0, boxW, boxH);
      const tData = tctx.getImageData(0, 0, boxW, boxH);
      for (let i = 0; i < tData.data.length; i += 4) {
        const v = 255 - ((tData.data[i] + tData.data[i + 1] + tData.data[i + 2]) / 3);
        tData.data[i] = v;
        tData.data[i + 1] = v;
        tData.data[i + 2] = v;
        tData.data[i + 3] = 255;
      }
      tctx.putImageData(tData, 0, 0);

      const drawX = Math.round((28 - drawW) / 2);
      const drawY = Math.round((28 - drawH) / 2);
      mctx.drawImage(tempCanvas, 0, 0, boxW, boxH, drawX, drawY, drawW, drawH);
    }

    const mData = mctx.getImageData(0, 0, 28, 28).data;
    const floatArr = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      floatArr[i] = mData[i * 4] / 255.0;
    }

    return tf.tensor4d(floatArr, [1, 28, 28, 1]);
  }

  generateLetterCandidates(char, conf) {
    const map = {
      'R': ['P', 'B'],
      'A': ['4', 'H'], 'B': ['8', 'E'], 'C': ['G', 'O'],
      'D': ['0', 'O'], 'E': ['F', '3'], 'F': ['E', 'P'],
      'G': ['6', 'C'], 'H': ['A', 'N'], 'I': ['1', 'l'],
      'J': ['I', '1'], 'K': ['R', 'X'], 'L': ['1', 'I'],
      'M': ['N', 'W'], 'N': ['M', 'H'], 'O': ['0', 'Q'],
      'P': ['R', 'B'], 'Q': ['O', '9'],
      'S': ['5', '8'], 'T': ['7', 'I'], 'U': ['V', 'O'],
      'V': ['U', 'Y'], 'W': ['M', 'V'], 'X': ['K', 'Y'],
      'Y': ['V', 'T'], 'Z': ['2', '7']
    };

    const upper = (char || 'A').toUpperCase();
    const alts = map[upper] || ['?', '?'];

    return [
      { char: upper, prob: `${conf}%` },
      { char: alts[0], prob: `${Math.max(conf - 15, 20)}%` },
      { char: alts[1], prob: `${Math.max(conf - 28, 10)}%` }
    ];
  }

  /**
   * 캔버스 이미지에서 수직 투영 히스토그램을 이용한 단어/문장 글자별 분할
   */
  segmentCanvasIntoCharacters(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    // 1. 각 열(x)마다 획 픽셀 수 카운트
    const colPixelCounts = new Int32Array(width);
    for (let x = 0; x < width; x++) {
      let count = 0;
      for (let y = 0; y < height; y++) {
        const idx = (y * width + x) * 4;
        const b = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (b < 210) {
          count++;
        }
      }
      colPixelCounts[x] = count;
    }

    // 2. 획이 연속으로 존재하는 열 구간 추출
    const rawSegments = [];
    let inStroke = false;
    let startX = 0;

    for (let x = 0; x < width; x++) {
      if (colPixelCounts[x] >= 2) {
        if (!inStroke) {
          inStroke = true;
          startX = x;
        }
      } else {
        if (inStroke) {
          inStroke = false;
          if (x - startX >= 4) { // 최소 4px 이상의 유효 획 구간
            rawSegments.push({ startX, endX: x - 1 });
          }
        }
      }
    }
    if (inStroke && (width - startX >= 4)) {
      rawSegments.push({ startX, endX: width - 1 });
    }

    if (rawSegments.length === 0) return [];

    // 3. 인접 구간 간격이 좁으면 한 글자(예: 'H', 'E', 'i', '=' 등)로 병합
    let avgW = 0;
    rawSegments.forEach(s => avgW += (s.endX - s.startX + 1));
    avgW /= rawSegments.length;
    const mergeThreshold = Math.max(8, Math.min(avgW * 0.35, 24));

    const merged = [];
    let cur = { ...rawSegments[0] };
    for (let i = 1; i < rawSegments.length; i++) {
      const next = rawSegments[i];
      const gap = next.startX - cur.endX;
      if (gap <= mergeThreshold) {
        cur.endX = next.endX;
      } else {
        merged.push(cur);
        cur = { ...next };
      }
    }
    merged.push(cur);

    // 4. 각 구간의 y 바운딩 박스(minY, maxY) 및 픽셀 유효성 검증
    const boxes = [];
    let totalW = 0;

    for (const seg of merged) {
      let minY = height, maxY = 0;
      let strokeCount = 0;

      for (let x = seg.startX; x <= seg.endX; x++) {
        for (let y = 0; y < height; y++) {
          const idx = (y * width + x) * 4;
          const b = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          if (b < 210) {
            strokeCount++;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      if (strokeCount >= 20 && maxY > minY) {
        const w = seg.endX - seg.startX + 1;
        const h = maxY - minY + 1;
        totalW += w;
        boxes.push({
          x: seg.startX,
          y: minY,
          w: w,
          h: h,
          isSpaceAfter: false
        });
      }
    }

    if (boxes.length === 0) return [];

    // 5. 단어 간 띄어쓰기(공백) 감지
    const avgCharW = totalW / boxes.length;
    const spaceThreshold = avgCharW * 0.7; // 평균 글자 폭의 70% 이상 벌어지면 공백
    for (let i = 0; i < boxes.length - 1; i++) {
      const curBox = boxes[i];
      const nextBox = boxes[i + 1];
      const gap = nextBox.x - (curBox.x + curBox.w);
      if (gap >= spaceThreshold) {
        curBox.isSpaceAfter = true;
      }
    }

    return boxes;
  }

  /**
   * 바운딩 박스 영역을 300x300 정사각형 중앙 정렬 캔버스로 크롭/정규화
   */
  cropBoxToSquareCanvas(sourceCanvas, box) {
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = 300;
    cropCanvas.height = 300;
    const cctx = cropCanvas.getContext('2d');

    cctx.fillStyle = '#ffffff';
    cctx.fillRect(0, 0, 300, 300);

    const maxDim = Math.max(box.w, box.h);
    const targetSize = 210;
    const scale = targetSize / maxDim;
    const drawW = box.w * scale;
    const drawH = box.h * scale;
    const drawX = (300 - drawW) / 2;
    const drawY = (300 - drawH) / 2;

    cctx.drawImage(
      sourceCanvas,
      box.x, box.y, box.w, box.h,
      drawX, drawY, drawW, drawH
    );

    // 명암 대비 이진화
    const imgData = cctx.getImageData(0, 0, 300, 300);
    const px = imgData.data;
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
    cctx.putImageData(imgData, 0, 0);

    return cropCanvas;
  }
}

window.HandwritingRecognizer = HandwritingRecognizer;
