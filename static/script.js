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
            const bubble = addAssistantMessage(data);
            if (data.answer) {
                await speakText(data.answer, bubble);
            }
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
// Audio recording
// ---------------------------
async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
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
        recordBtn.textContent = 'Arrêter';
        recordBtn.classList.add('recording');

        // Auto-stop at the max duration so a user can't record indefinitely.
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
    recordBtn.textContent = 'Enregistrer';
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

    // Stash this turn's payload on the element so tab switches don't refetch.
    bubble._turnData = {
        turnId: data.turn_id,
        answer: data.answer || '',
        sql: data.sql || '',
        columns: data.columns || [],
        rows: data.rows || [],
        graphFigure: null,
        graphError: null
    };

    const hasSql = data.action === 'new_query' || data.action === 'follow_up';

    if (hasSql) {
        const tabs = document.createElement('div');
        tabs.className = 'message-turn-tabs';
        bubble._activeView = 'answer';

        const views = [
            { key: 'answer', label: 'Réponse' },
            { key: 'table', label: 'Tableau SQL' },
            { key: 'graph', label: 'Graphique' }
        ];
        views.forEach((v, i) => {
            const btn = document.createElement('button');
            btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
            btn.textContent = v.label;
            btn.dataset.view = v.key;
            btn.addEventListener('click', () => {
                tabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                bubble._activeView = v.key;
                renderTurnView(bubble, v.key);
            });
            tabs.appendChild(btn);
        });
        bubble.appendChild(tabs);

        const content = document.createElement('div');
        content.className = 'turn-content';
        bubble.appendChild(content);

        renderTurnView(bubble, 'answer');
    } else {
        const content = document.createElement('div');
        content.className = 'turn-content';
        content.textContent = data.answer || '';
        bubble.appendChild(content);
    }

    chat.appendChild(bubble);
    scrollChatToBottom();
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

    // Already fetched: just redraw from the cached figure, no new request.
    if (turnData.graphFigure) {
        drawPlotlyFigure(content, bubble.id, turnData.graphFigure);
        return;
    }
    if (turnData.graphError) {
        showGraphMessage(content, turnData.graphError);
        return;
    }

    showGraphMessage(content, 'Génération du graphique...');

    try {
        const response = await fetch('/graph', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ turn_id: turnData.turnId })
        });
        const data = await response.json();

        // The user might have switched to a different tab (or cleared the
        // conversation) while this fetch was in flight. The content div is
        // reused across tab switches, so check the active tab explicitly
        // rather than relying on whether the node is still in the document.
        if (data.error) {
            turnData.graphError = data.error;
            if (bubble._activeView === 'graph') showGraphMessage(content, data.error);
            return;
        }

        turnData.graphFigure = data.figure;
        if (bubble._activeView === 'graph') drawPlotlyFigure(content, bubble.id, data.figure);
    } catch (err) {
        const message = 'Erreur réseau lors de la génération du graphique.';
        turnData.graphError = message;
        if (bubble._activeView === 'graph') showGraphMessage(content, message);
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

async function speakText(text, bubble) {
    if (!text) return;
    try {
        const response = await fetch('/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        if (!response.ok) {
            const err = await response.json();
            console.error('TTS error:', err.error);
            return;
        }
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        try {
            await audio.play();
        } catch (playErr) {
            // The browser blocked autoplay (this can happen once the
            // original click's "user gesture" trust has expired after a
            // couple of awaited fetch calls). Offer a manual play button
            // instead of failing with no feedback.
            console.warn('Autoplay blocked, offering manual playback:', playErr);
            if (bubble) {
                addManualPlayButton(bubble, audioUrl);
            }
        }
    } catch (err) {
        console.error('Error playing audio:', err);
    }
}

function addManualPlayButton(bubble, audioUrl) {
    const btn = document.createElement('button');
    btn.className = 'tab-btn';
    btn.style.marginTop = '8px';
    btn.textContent = 'Écouter la réponse';
    btn.addEventListener('click', () => {
        const audio = new Audio(audioUrl);
        audio.play().catch(err => console.error('Playback failed:', err));
    });
    bubble.appendChild(btn);
}
