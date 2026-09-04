// Global variables for audio recording
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordingAutoStopTimer = null;
let recordingCountdownInterval = null;

// Must match MAX_AUDIO_DURATION_SECONDS in config.py.
const MAX_RECORDING_SECONDS = 30;

// Used to give each assistant turn's DOM element a unique id.
let turnCounter = 0;

// ---------------------------
// Text submission
// ---------------------------
async function submitText() {
    const input = document.getElementById('textInput');
    const question = input.value.trim();
    if (!question) {
        showError('Veuillez entrer une question.');
        return;
    }

    input.value = '';
    clearError();
    addUserMessage(question);
    showLoading(true);
    setInputEnabled(false);

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        if (data.error) {
            showError(data.error);
        } else {
            addAssistantMessage(data);
        }
    } catch (err) {
        showError('Erreur réseau: ' + err.message);
    } finally {
        showLoading(false);
        setInputEnabled(true);
    }
}

function setInputEnabled(enabled) {
    document.getElementById('textInput').disabled = !enabled;
    document.querySelector('button[onclick="submitText()"]').disabled = !enabled;
}

// ---------------------------
// New conversation
// ---------------------------
async function startNewConversation() {
    try {
        await fetch('/clear', { method: 'POST' });
    } catch (err) {
        showError('Erreur lors de la réinitialisation: ' + err.message);
        return;
    }
    document.getElementById('chatContainer').innerHTML = '';
    document.getElementById('textInput').value = '';
    clearError();
}

// ---------------------------
// Audio recording: hold to record, release to stop.
// ---------------------------
function setupRecordButton() {
    const recordBtn = document.getElementById('recordBtn');

    const start = (e) => { e.preventDefault(); if (!isRecording) startRecording(); };
    const stop = (e) => { e.preventDefault(); if (isRecording) stopRecording(); };

    recordBtn.addEventListener('mousedown', start);
    recordBtn.addEventListener('touchstart', start, { passive: false });
    recordBtn.addEventListener('mouseup', stop);
    recordBtn.addEventListener('mouseleave', stop);
    recordBtn.addEventListener('touchend', stop);
    recordBtn.addEventListener('touchcancel', stop);
    // Basic keyboard accessibility (hold Enter/Space while focused).
    recordBtn.addEventListener('keydown', (e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !e.repeat) start(e);
    });
    recordBtn.addEventListener('keyup', (e) => {
        if (e.key === 'Enter' || e.key === ' ') stop(e);
    });
}

async function startRecording() {
    const recordBtn = document.getElementById('recordBtn');
    const status = document.getElementById('recordingStatus');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(track => track.stop());
            const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
            try {
                showLoading(true);
                const wavBlob = await convertWebmToWav(webmBlob);
                await sendAudioToBackend(wavBlob);
            } catch (err) {
                showError('Erreur lors de la conversion audio: ' + err.message);
            } finally {
                showLoading(false);
            }
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add('recording');

        // Auto-stop at the max duration even if still held, as a safety net.
        let remaining = MAX_RECORDING_SECONDS;
        status.textContent = `Enregistrement en cours... (max ${remaining}s)`;
        recordingCountdownInterval = setInterval(() => {
            remaining -= 1;
            if (remaining > 0) {
                status.textContent = `Enregistrement en cours... (${remaining}s restantes)`;
            }
        }, 1000);
        recordingAutoStopTimer = setTimeout(() => {
            stopRecording();
        }, MAX_RECORDING_SECONDS * 1000);
    } catch (err) {
        console.error('Erreur microphone:', err);
        status.textContent = 'Erreur: ' + err.message;
    }
}

function stopRecording() {
    const recordBtn = document.getElementById('recordBtn');
    const status = document.getElementById('recordingStatus');

    if (recordingAutoStopTimer) {
        clearTimeout(recordingAutoStopTimer);
        recordingAutoStopTimer = null;
    }
    if (recordingCountdownInterval) {
        clearInterval(recordingCountdownInterval);
        recordingCountdownInterval = null;
    }

    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
    }
    isRecording = false;
    recordBtn.classList.remove('recording');
    status.textContent = '';
}

async function convertWebmToWav(webmBlob) {
    const arrayBuffer = await webmBlob.arrayBuffer();
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContextCtor();
    try {
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        return encodeWavBlob(audioBuffer);
    } finally {
        audioCtx.close();
    }
}

function encodeWavBlob(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const bitDepth = 16;

    const channelData = [];
    for (let ch = 0; ch < numChannels; ch++) {
        channelData.push(audioBuffer.getChannelData(ch));
    }
    const interleaved = numChannels === 2
        ? interleaveStereo(channelData[0], channelData[1])
        : channelData[0];

    const bytesPerSample = bitDepth / 8;
    const dataLength = interleaved.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    writeAsciiString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeAsciiString(view, 8, 'WAVE');
    writeAsciiString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);            // fmt chunk size
    view.setUint16(20, 1, true);             // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
    view.setUint16(32, numChannels * bytesPerSample, true);
    view.setUint16(34, bitDepth, true);
    writeAsciiString(view, 36, 'data');
    view.setUint32(40, dataLength, true);

    floatTo16BitPCM(view, 44, interleaved);

    return new Blob([view], { type: 'audio/wav' });
}

function interleaveStereo(left, right) {
    const length = left.length + right.length;
    const result = new Float32Array(length);
    let index = 0;
    for (let i = 0; i < left.length; i++) {
        result[index++] = left[i];
        result[index++] = right[i];
    }
    return result;
}

function writeAsciiString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

function floatTo16BitPCM(view, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, input[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
}

async function sendAudioToBackend(wavBlob) {
    const formData = new FormData();
    formData.append('audio', wavBlob, 'recording.wav');

    try {
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.text !== undefined) {
            document.getElementById('textInput').value = data.text;
            if (!data.text) {
                showError('Aucune parole détectée, réessayez.');
            }
        } else {
            showError('Erreur de transcription: ' + (data.error || 'inconnue'));
        }
    } catch (err) {
        showError('Erreur réseau: ' + err.message);
    }
}

// ---------------------------
// Chat rendering
// ---------------------------
function addUserMessage(text) {
    const chat = document.getElementById('chatContainer');
    const bubble = document.createElement('div');
    bubble.className = 'message user';
    bubble.textContent = text;
    chat.appendChild(bubble);
    scrollChatToBottom();
}

function addAssistantMessage(data) {
    const chat = document.getElementById('chatContainer');
    const bubble = document.createElement('div');
    bubble.className = 'message assistant';
    bubble.id = 'turn-' + (turnCounter++);

    const hasSql = data.action === 'new_query' || data.action === 'follow_up';
    const chartWorthy = hasSql && !!data.chart_worthy;
    // Always default to the answer: it's already fully ready with zero
    // wait, unlike the graph, which still needs its own LLM call. Never
    // open a turn on a loading spinner when a complete answer is sitting
    // right there.
    const defaultView = 'answer';

    bubble._turnData = {
        turnId: data.turn_id,
        answer: data.answer || '',
        sql: data.sql || '',
        columns: data.columns || [],
        rows: data.rows || [],
        graphFigure: null,
        graphError: null,
        graphFetchStarted: false,
        audioUrl: null
    };
    bubble._activeView = defaultView;

    const toolbar = document.createElement('div');
    toolbar.className = 'message-toolbar';

    // Every assistant message gets the same three tabs; ones that don't
    // apply to this particular turn are blurred rather than hidden, so
    // the toolbar shape stays consistent across the whole conversation.
    const views = [
        { key: 'answer', label: 'Réponse', relevant: true },
        { key: 'table', label: 'Tableau SQL', relevant: hasSql },
        { key: 'graph', label: 'Graphique', relevant: chartWorthy }
    ];
    views.forEach((v) => {
        const btn = document.createElement('button');
        btn.className = 'tab-btn' + (v.key === defaultView ? ' active' : '');
        btn.textContent = v.label;
        btn.dataset.view = v.key;
        if (!v.relevant) {
            btn.classList.add('blurred');
            btn.title = v.key === 'graph'
                ? "Pas de graphique pertinent pour ce résultat."
                : "Aucune donnée SQL pour cet échange.";
        } else {
            btn.addEventListener('click', () => {
                toolbar.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                bubble._activeView = v.key;
                renderTurnView(bubble, v.key);
            });
        }
        toolbar.appendChild(btn);
    });

    const speakerBtn = document.createElement('button');
    speakerBtn.className = 'speaker-btn';
    speakerBtn.textContent = '🔊';
    speakerBtn.title = 'Écouter la réponse';
    speakerBtn.addEventListener('click', () => playAnswerAudio(bubble, speakerBtn));
    toolbar.appendChild(speakerBtn);

    bubble.appendChild(toolbar);

    const content = document.createElement('div');
    content.className = 'turn-content';
    bubble.appendChild(content);

    renderTurnView(bubble, defaultView);

    chat.appendChild(bubble);
    scrollChatToBottom();

    if (chartWorthy) {
        fetchGraphData(bubble);  // background only, does not touch the visible tab
    }

    return bubble;
}

function renderTurnView(bubble, view) {
    const content = bubble.querySelector('.turn-content');
    const { answer, sql, columns, rows } = bubble._turnData;

    content.innerHTML = '';

    if (view === 'answer') {
        content.textContent = answer;
        return;
    }

    if (view === 'table') {
        if (sql) {
            const sqlBox = document.createElement('div');
            sqlBox.className = 'sql-box';
            sqlBox.innerHTML = '<strong>SQL générée :</strong><br>' + escapeHtml(sql);
            content.appendChild(sqlBox);
        }
        if (columns.length > 0) {
            content.appendChild(buildTable(columns, rows));
        } else {
            const p = document.createElement('p');
            p.textContent = 'Aucune donnée retournée.';
            content.appendChild(p);
        }
        return;
    }

    if (view === 'graph') {
        renderGraphView(bubble, content);
    }
}

async function renderGraphView(bubble, content) {
    const turnData = bubble._turnData;

    // Already fetched (by an earlier click, or the background prefetch
    // that runs for chart-worthy turns): redraw from cache, no new request.
    if (turnData.graphFigure) {
        drawPlotlyFigure(content, bubble.id, turnData.graphFigure);
        return;
    }
    if (turnData.graphError) {
        showGraphMessage(content, turnData.graphError);
        return;
    }

    showGraphMessage(content, 'Génération du graphique...');
    await fetchGraphData(bubble);
    // fetchGraphData re-renders itself if we're still on the graph tab
    // once it resolves (see below), nothing more to do here.
}

async function fetchGraphData(bubble) {
    // Fetches (once) and caches the chart for this turn. Safe to call
    // proactively in the background: does not touch the DOM unless the
    // user is actually looking at the graph tab when it resolves, so a
    // silent prefetch for a chart-worthy result can never interrupt
    // someone reading the answer.
    const turnData = bubble._turnData;
    if (turnData.graphFigure || turnData.graphError || turnData.graphFetchStarted) return;
    turnData.graphFetchStarted = true;

    try {
        const response = await fetch('/graph', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ turn_id: turnData.turnId })
        });
        const data = await response.json();

        if (data.error) {
            turnData.graphError = data.error;
        } else {
            turnData.graphFigure = data.figure;
        }
    } catch (err) {
        turnData.graphError = 'Erreur réseau lors de la génération du graphique.';
    }

    // The content div is reused across tab switches, so only touch it if
    // the user is currently actually looking at the graph tab, whether
    // because they clicked it before this resolved, or clicked it after a
    // background prefetch was already in flight.
    if (bubble._activeView === 'graph') {
        const content = bubble.querySelector('.turn-content');
        if (turnData.graphFigure) {
            drawPlotlyFigure(content, bubble.id, turnData.graphFigure);
        } else {
            showGraphMessage(content, turnData.graphError);
        }
    }
}

function showGraphMessage(content, message) {
    content.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'graph-placeholder';
    p.textContent = message;
    content.appendChild(p);
}

function drawPlotlyFigure(content, bubbleId, figure) {
    content.innerHTML = '';
    const div = document.createElement('div');
    div.id = 'chart-' + bubbleId;
    div.style.width = '100%';
    div.style.minHeight = '340px';
    content.appendChild(div);
    Plotly.newPlot(div, figure.data, figure.layout, { responsive: true, displaylogo: false });
}

function buildTable(columns, rows) {
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

function scrollChatToBottom() {
    const chat = document.getElementById('chatContainer');
    chat.scrollTop = chat.scrollHeight;
}

// ---------------------------
// Text-to-speech: click-to-play per message. Deliberately manual rather
// than auto-playing on arrival, since a direct click is always a valid
// user gesture and never gets blocked by the browser's autoplay policy.
// ---------------------------
async function playAnswerAudio(bubble, btn) {
    const turnData = bubble._turnData;
    if (!turnData.answer) return;

    if (turnData.audioUrl) {
        new Audio(turnData.audioUrl).play().catch(err => console.error('Playback failed:', err));
        return;
    }

    const originalLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = '...';

    try {
        const response = await fetch('/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: turnData.answer })
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            console.error('TTS error:', err.error);
            showError("Impossible de générer l'audio pour cette réponse.");
            return;
        }
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        turnData.audioUrl = audioUrl;
        await new Audio(audioUrl).play().catch(err => console.error('Playback failed:', err));
    } catch (err) {
        console.error('Error fetching audio:', err);
        showError("Erreur réseau lors de la génération de l'audio.");
    } finally {
        btn.disabled = false;
        btn.textContent = originalLabel;
    }
}

// ---------------------------
// Misc helpers
// ---------------------------
function showError(message) {
    const errorDiv = document.getElementById('errorDisplay');
    if (errorDiv) {
        errorDiv.textContent = message;
    }
}

function clearError() {
    showError('');
}

function showLoading(isLoading) {
    const loadingDiv = document.getElementById('loadingIndicator');
    if (loadingDiv) {
        loadingDiv.style.display = isLoading ? 'block' : 'none';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

setupRecordButton();
