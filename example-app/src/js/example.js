import { FaceRecognition } from 'capacitor-face-recognition-tflite';

// ============================
// State
// ============================
let currentFrameBase64 = null;   // frame terakhir (base64 murni, tanpa prefix data URL)
let lastTest = null;             // hasil uji terakhir untuk dataset logger
let yawStreamTimer = null;
const datasetLog = [];

// ============================
// Helper
// ============================
const $ = (id) => document.getElementById(id);

function dataUrlToBase64(dataUrl) {
  return dataUrl.split(',')[1];
}

function setSingleResult(html, json) {
  $('singleResult').innerHTML = html || '';
  $('singleJson').textContent = json ? JSON.stringify(json, null, 2) : 'Belum ada hasil.';
}

function setSessionResult(html, json) {
  $('sessionResult').innerHTML = html || '';
  $('sessionJson').textContent = json ? JSON.stringify(json, null, 2) : 'Belum ada hasil.';
}

function badge(text, kind) {
  return `<span class="badge ${kind}">${text}</span>`;
}

function requireFrame() {
  if (!currentFrameBase64) {
    alert('Ambil frame dulu (atau muat file gambar).');
    return false;
  }
  return true;
}

// ============================
// Kamera
// ============================
let mediaStream = null;

async function startCamera() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  $('video').srcObject = mediaStream;
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
    $('video').srcObject = null;
  }
  stopYawStream();
}

function captureFrame() {
  const video = $('video');
  if (!video.videoWidth) {
    alert('Kamera belum aktif.');
    return;
  }
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  currentFrameBase64 = dataUrlToBase64(canvas.toDataURL('image/jpeg', 0.85));
  setSingleResult(badge(`Frame diambil (${canvas.width}x${canvas.height})`, 'ok'), null);
}

function loadFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    currentFrameBase64 = dataUrlToBase64(reader.result);
    setSingleResult(badge(`File dimuat: ${file.name}`, 'ok'), null);
  };
  reader.readAsDataURL(file);
}

// ============================
// LivenessConfig dari form
// ============================
function collectConfig() {
  return {
    liveThreshold: parseFloat($('liveThreshold').value),
    spoofVeto: parseFloat($('spoofVeto').value),
    minFaceRatio: parseFloat($('minFaceRatio').value),
    blurMin: parseFloat($('blurMin').value),
    maxPose: parseFloat($('maxPose').value),
    edgeMargin: parseFloat($('edgeMargin').value),
    exposureMin: parseFloat($('exposureMin').value),
    exposureMax: parseFloat($('exposureMax').value),
    minFrameSide: parseInt($('minFrameSide').value, 10),
    yawSign: parseInt($('yawSign').value, 10),
    debug: $('debug').value === 'true',
  };
}

// ============================
// Uji Single Frame
// ============================
async function runCheckLiveness() {
  if (!requireFrame()) return;
  try {
    const result = await FaceRecognition.checkLiveness({ imageBase64: currentFrameBase64 });
    const html = `
      ${badge(result.label, result.isLive ? 'ok' : 'fail')}
      ${badge(result.isLive ? 'LIVE' : 'SPOOF', result.isLive ? 'ok' : 'fail')}
      liveScore=${result.liveScore.toFixed(3)} printScore=${result.printScore.toFixed(3)}
      replayScore=${result.replayScore.toFixed(3)} (${result.thresholdVersion})`;
    setSingleResult(html, result);
    lastTest = { method: 'checkLiveness', isLive: result.isLive, result };
  } catch (e) {
    setSingleResult(badge('ERROR', 'fail'), { error: String(e) });
  }
}

async function runVerifyFace() {
  if (!requireFrame()) return;
  try {
    const result = await FaceRecognition.verifyFace({
      imageBase64: currentFrameBase64,
      config: collectConfig(),
    });
    const q = result.quality;
    let html = `${badge(q.passed ? 'QUALITY OK' : `QUALITY: ${q.reason}`, q.passed ? 'ok' : 'fail')} `;
    if (result.liveness) {
      html += `${badge(result.liveness.label, result.isLive ? 'ok' : 'fail')}
        liveScore=${result.liveness.liveScore.toFixed(3)}
        print=${result.liveness.printScore.toFixed(3)} replay=${result.liveness.replayScore.toFixed(3)}
        blur=${q.blur.toFixed(1)} faceRatio=${q.faceRatio.toFixed(2)}`;
    }
    if (result.match) {
      html += `<br/>match: ${badge(result.match.isMatch ? 'MATCH' : 'NO MATCH', result.match.isMatch ? 'ok' : 'fail')}
        score=${result.match.score.toFixed(3)} (threshold ${result.match.threshold})`;
    }
    setSingleResult(html, result);
    lastTest = { method: 'verifyFace', isLive: result.isLive, result };
  } catch (e) {
    setSingleResult(badge('ERROR', 'fail'), { error: String(e) });
  }
}

async function runDetectFaces() {
  if (!requireFrame()) return;
  try {
    const result = await FaceRecognition.detectFaces({ imageBase64: currentFrameBase64 });
    const faces = result.faces || [];
    const html = `${badge(`${result.count} wajah`, result.count === 1 ? 'ok' : 'warn')}
      ${faces.map((f) =>
        `yaw=${f.headEulerAngleY.toFixed(1)} pitch=${f.headEulerAngleX.toFixed(1)} roll=${f.headEulerAngleZ.toFixed(1)}`
      ).join('<br/>')}`;
    setSingleResult(html, result);
  } catch (e) {
    setSingleResult(badge('ERROR', 'fail'), { error: String(e) });
  }
}

// ============================
// Yaw Sign Tester
// ============================
async function yawTick() {
  try {
    const canvas = document.createElement('canvas');
    const video = $('video');
    if (!video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const frame = dataUrlToBase64(canvas.toDataURL('image/jpeg', 0.7));
    const result = await FaceRecognition.detectFaces({ imageBase64: frame });
    const f = result.faces && result.faces[0];
    if (f) {
      $('yawValue').textContent = f.headEulerAngleY.toFixed(1);
      $('pitchValue').textContent = f.headEulerAngleX.toFixed(1);
      $('rollValue').textContent = f.headEulerAngleZ.toFixed(1);
    }
  } catch (e) {
    // diamkan; tick berikutnya akan mencoba lagi
  }
}

function toggleYawStream() {
  if (yawStreamTimer) {
    stopYawStream();
  } else {
    yawStreamTimer = setInterval(yawTick, 500);
    $('btnYawStream').textContent = 'Stop Stream Angles';
    $('btnYawStream').classList.add('warn');
  }
}

function stopYawStream() {
  if (yawStreamTimer) {
    clearInterval(yawStreamTimer);
    yawStreamTimer = null;
  }
  const btn = $('btnYawStream');
  if (btn) {
    btn.textContent = 'Mulai Stream Angles';
    btn.classList.remove('warn');
  }
}

// ============================
// Sesi Multi-frame (challenge)
// ============================
async function runSession() {
  try {
    // 1. Dapatkan challenge acak
    const { challenge } = await FaceRecognition.getLivenessChallenge();
    const instruction = challenge === 'LOOK_LEFT'
      ? 'TOLEH KE KIRI perlahan, lalu kembali lurus.'
      : 'TOLEH KE KANAN perlahan, lalu kembali lurus.';
    $('challengeHint').innerHTML = `<b>Challenge: ${challenge}</b> — ${instruction} Mengambil 5 frame...`;
    setSessionResult(badge(`Challenge: ${challenge}`, 'warn'), null);

    // 2. Ambil 5 frame dalam ~1,5 detik (300ms interval)
    const video = $('video');
    if (!video.videoWidth) {
      alert('Kamera belum aktif.');
      return;
    }
    const frames = [];
    for (let i = 0; i < 5; i++) {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      frames.push(dataUrlToBase64(canvas.toDataURL('image/jpeg', 0.85)));
      if (i < 4) await new Promise((r) => setTimeout(r, 300));
    }

    // 3. Verifikasi sesi
    const result = await FaceRecognition.verifyFaceSequence({
      images: frames,
      challenge,
      config: collectConfig(),
    });
    const html = `
      ${badge(result.passed ? 'SESI LOLOS' : `SESI GAGAL: ${result.reason}`, result.passed ? 'ok' : 'fail')}
      medianLiveScore=${result.medianLiveScore.toFixed(3)}
      liveVotes=${result.liveVotes}/${result.frameCount}
      yaw=[${result.minYaw.toFixed(1)}..${result.maxYaw.toFixed(1)}]
      frontalFrames=${result.frontalFrames}
      identityConsistent=${result.identityConsistent}`;
    setSessionResult(html, result);
    lastTest = { method: 'verifyFaceSequence', isLive: result.passed, result };
    $('challengeHint').textContent =
      'Tekan mulai, ikuti instruksi challenge yang muncul. 5 frame diambil otomatis dalam ~1,5 detik.';
  } catch (e) {
    setSessionResult(badge('ERROR', 'fail'), { error: String(e) });
  }
}

async function runNativeSession() {
  try {
    setSessionResult(badge('Membuka kamera native (frame tidak melewati JS)...', 'warn'), null);
    const result = await FaceRecognition.captureLivenessSession({
      config: collectConfig(),
    });
    const integrity = result.deviceIntegrity || {};
    const html = `
      ${badge(result.passed ? 'SESI LOLOS' : `SESI GAGAL: ${result.reason}`, result.passed ? 'ok' : 'fail')}
      ${badge('NATIVE CAPTURE', 'warn')}
      medianLiveScore=${result.medianLiveScore.toFixed(3)}
      liveVotes=${result.liveVotes}/${result.frameCount}
      <br/>deviceIntegrity: emulator=${integrity.isEmulator} rooted=${integrity.isRooted}
      debuggable=${integrity.isDebuggable} hooks=${integrity.hooksDetected}
      <br/>auditId=${result.auditId ?? '-'}`;
    setSessionResult(html, result);
    lastTest = { method: 'captureLivenessSession', isLive: result.passed, result };
  } catch (e) {
    setSessionResult(badge('ERROR', 'fail'), { error: String(e) });
  }
}

// ============================
// Dataset Logger
// ============================
function labelLast(kind) {
  if (!lastTest) {
    alert('Belum ada hasil uji yang bisa dilabeli.');
    return;
  }
  datasetLog.push({
    timestamp: new Date().toISOString(),
    kind: lastTest.method,
    label: kind,
    isLive: lastTest.isLive,
    liveScore: lastTest.result.liveScore ?? lastTest.result.medianLiveScore ?? null,
    raw: lastTest.result,
  });
  renderLog();
}

function renderLog() {
  const tbody = $('logTable').querySelector('tbody');
  tbody.innerHTML = datasetLog.map((entry, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${entry.kind}</td>
      <td>${entry.label}</td>
      <td>${entry.liveScore != null ? entry.liveScore.toFixed(3) : '-'}</td>
      <td>${entry.isLive ? 'lolos live' : 'ditolak'}</td>
    </tr>`).join('');

  // Estimasi APCER / BPCER sederhana dari log
  const live = datasetLog.filter((e) => e.label === 'LIVE');
  const spoof = datasetLog.filter((e) => e.label !== 'LIVE');
  const liveRejected = live.filter((e) => !e.isLive).length;
  const spoofPassed = spoof.filter((e) => e.isLive).length;
  const bpcer = live.length ? ((liveRejected / live.length) * 100).toFixed(1) : '-';
  const apcer = spoof.length ? ((spoofPassed / spoof.length) * 100).toFixed(1) : '-';
  $('datasetStats').innerHTML = `
    Sampel: LIVE=${live.length}, REPLAY=${datasetLog.filter((e) => e.label === 'REPLAY').length},
    PRINT=${datasetLog.filter((e) => e.label === 'PRINT').length}<br/>
    Estimasi BPCER (live ditolak): <b>${bpcer}%</b> &nbsp;
    Estimasi APCER (spoof lolos): <b>${apcer}%</b>
    <br/><span class="hint">Angka sementara — laporan resmi per jenis serangan, dengan set kalibrasi/uji terpisah (lihat plan Fase 5).</span>`;
}

function exportJson() {
  downloadFile('dataset-log.json', JSON.stringify(datasetLog, null, 2), 'application/json');
}

function exportCsv() {
  const header = 'timestamp,kind,label,isLive,liveScore';
  const rows = datasetLog.map((e) =>
    [e.timestamp, e.kind, e.label, e.isLive, e.liveScore ?? ''].join(','));
  downloadFile('dataset-log.csv', [header, ...rows].join('\n'), 'text/csv');
}

function downloadFile(name, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

// ============================
// Wiring
// ============================
const RANGE_FIELDS = [
  'liveThreshold', 'spoofVeto', 'minFaceRatio', 'blurMin',
  'maxPose', 'edgeMargin', 'exposureMin', 'exposureMax',
];
RANGE_FIELDS.forEach((id) => {
  const el = $(id);
  el.addEventListener('input', () => {
    const target = $(`v_${id}`);
    if (target) target.textContent = el.value;
  });
});

$('btnStartCamera').addEventListener('click', startCamera);
$('btnStopCamera').addEventListener('click', stopCamera);
$('btnCapture').addEventListener('click', captureFrame);
$('fileInput').addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) loadFile(e.target.files[0]);
});
$('btnYawStream').addEventListener('click', toggleYawStream);
$('btnCheckLiveness').addEventListener('click', runCheckLiveness);
$('btnVerifyFace').addEventListener('click', runVerifyFace);
$('btnDetectFaces').addEventListener('click', runDetectFaces);
$('btnSession').addEventListener('click', runSession);
$('btnNativeSession').addEventListener('click', runNativeSession);
$('btnLabelLive').addEventListener('click', () => labelLast('LIVE'));
$('btnLabelReplay').addEventListener('click', () => labelLast('REPLAY'));
$('btnLabelPrint').addEventListener('click', () => labelLast('PRINT'));
$('btnExportJson').addEventListener('click', exportJson);
$('btnExportCsv').addEventListener('click', exportCsv);
$('btnClearLog').addEventListener('click', () => {
  datasetLog.length = 0;
  renderLog();
});
