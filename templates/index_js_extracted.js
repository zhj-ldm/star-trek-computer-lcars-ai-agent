
// ══════════════════════════════════════════════
// THEME
// ══════════════════════════════════════════════
const THEME_KEY = 'marvis-chat-theme';

function isLcars() { return document.body.classList.contains('lcars-theme'); }

function applyTheme(theme) {
  if (theme === 'lcars') {
    document.body.classList.add('lcars-theme');
  } else {
    document.body.classList.remove('lcars-theme');
  }
  localStorage.setItem(THEME_KEY, theme);
  refreshAllUI();
}

function toggleTheme() {
  applyTheme(document.getElementById('themeToggle').checked ? 'lcars' : 'apple');
}

(function initTheme() {
  if (localStorage.getItem(THEME_KEY) === 'lcars') {
    document.body.classList.add('lcars-theme');
  }
})();

// ══════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════
let currentConvId = null;
let configCache = null;

// ══════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

let toastTimer;
function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function(){ el.classList.remove('show'); }, 2000);
}

// ══════════════════════════════════════════════
// APPLE THEME OPS
// ══════════════════════════════════════════════
function renderAppleConversations(convs) {
  const list = document.getElementById('apple-conv-list');
  list.innerHTML = '';
  convs.forEach(function(c) {
    const div = document.createElement('div');
    div.className = 'conv-item' + (c.id === currentConvId ? ' active' : '');
    div.onclick = function() { switchConversation(c.id); };
    div.innerHTML = '<span class="conv-title" title="' + esc(c.title) + '">' + esc(c.title) + '</span>' +
      '<button class="conv-delete" onclick="event.stopPropagation();deleteConversation(\'' + c.id + '\')">&times;</button>';
    list.appendChild(div);
  });
}

function clearAppleChat() {
  document.getElementById('apple-chat-area').innerHTML = '<div class="empty-state"><div class="icon">💬</div><div>开始一段新对话</div></div>';
}

function renderAppleMessages(msgs) {
  const area = document.getElementById('apple-chat-area');
  area.innerHTML = '';
  if (!msgs.length) { clearAppleChat(); return; }
  msgs.forEach(function(m) {
    const div = document.createElement('div');
    div.className = 'message ' + m.role;
    div.textContent = m.content;
    area.appendChild(div);
  });
  area.scrollTop = area.scrollHeight;
}

function addAppleMessage(role, content) {
  const area = document.getElementById('apple-chat-area');
  const empty = area.querySelector('.empty-state');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.textContent = content;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

function showAppleTyping() {
  const area = document.getElementById('apple-chat-area');
  const empty = area.querySelector('.empty-state');
  if (empty) empty.remove();
  const indicator = document.createElement('div');
  indicator.className = 'typing-indicator';
  indicator.id = 'apple-typing';
  indicator.innerHTML = '<span></span><span></span><span></span>';
  area.appendChild(indicator);
  area.scrollTop = area.scrollHeight;
}

function hideAppleTyping() {
  const el = document.getElementById('apple-typing');
  if (el) el.remove();
}

function handleAppleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendAppleMessage(); }
}

async function sendAppleMessage() {
  const text = document.getElementById('apple-input').value.trim();
  if (!text) return;
  const btn = document.getElementById('apple-send-btn');
  btn.disabled = true;
  document.getElementById('apple-input').value = '';
  addAppleMessage('user', text);
  showAppleTyping();
  await doChat(text, function(reply, err) {
    hideAppleTyping();
    if (err) { addAppleMessage('ai', 'Error: ' + err); }
    else { addAppleMessage('ai', reply); }
    btn.disabled = false;
  });
}

// ══════════════════════════════════════════════
// LCARS THEME OPS
// ══════════════════════════════════════════════
const LCARS_COLORS = ['lcars-golden-tanoi-bg','lcars-neon-carrot-bg','lcars-mariner-bg','lcars-dodger-blue-bg','lcars-danub-bg','lcars-lilac-bg','lcars-hopbush-bg','lcars-atomic-tangerine-bg','lcars-bourbon-bg','lcars-rust-bg','lcars-sandy-brown-bg','lcars-red-damask-bg','lcars-blue-bell-bg'];

function renderLcarsConversations(convs) {
  const list = document.getElementById('lcars-conv-list');
  list.innerHTML = '';
  convs.forEach(function(c, i) {
    const color = LCARS_COLORS[i % LCARS_COLORS.length];
    const div = document.createElement('div');
    div.className = 'lcars-conv-item ' + color + ' lcars-element button';
    if (c.id === currentConvId) div.classList.add('active');
    div.onclick = function() { switchConversation(c.id); };
    div.innerHTML = '<span class="lcars-conv-title" title="' + esc(c.title) + '">' + esc(c.title.substring(0, 14)) + '</span>' +
      '<button class="lcars-conv-delete" onclick="event.stopPropagation();deleteConversation(\'' + c.id + '\')">X</button>';
    list.appendChild(div);
  });
  document.getElementById('lcars-msg-count').textContent = convs.length + ' CHAT';
}

function clearLcarsChat() {
  document.getElementById('lcars-chat-area').innerHTML = '<div id="lcars-empty" style="flex:1;display:flex;align-items:center;justify-content:center;color:#f90;font-size:14px;text-transform:uppercase;"><span>READY &mdash; ENTER MESSAGE BELOW</span></div>';
}

function renderLcarsMessages(msgs) {
  const area = document.getElementById('lcars-chat-area');
  area.innerHTML = '';
  if (!msgs.length) { clearLcarsChat(); return; }
  msgs.forEach(function(m) {
    const div = document.createElement('div');
    div.className = 'lcars-chat-msg ' + m.role;
    div.textContent = m.content;
    area.appendChild(div);
  });
  area.scrollTop = area.scrollHeight;
}

function addLcarsMessage(role, content) {
  const area = document.getElementById('lcars-chat-area');
  const empty = document.getElementById('lcars-empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'lcars-chat-msg ' + role;
  div.textContent = content;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
  updateLcarsStatus();
}

function showLcarsTyping() {
  const area = document.getElementById('lcars-chat-area');
  const empty = document.getElementById('lcars-empty');
  if (empty) empty.remove();
  const indicator = document.createElement('div');
  indicator.className = 'lcars-typing';
  indicator.id = 'lcars-typing';
  indicator.innerHTML = '<span></span><span></span><span></span>';
  area.appendChild(indicator);
  area.scrollTop = area.scrollHeight;
}

function hideLcarsTyping() {
  const el = document.getElementById('lcars-typing');
  if (el) el.remove();
}

function updateLcarsStatus() {
  const msgs = document.querySelectorAll('#lcars-chat-area .lcars-chat-msg');
  document.getElementById('lcars-msg-count').textContent = msgs.length + ' MSG';
}

function handleLcarsKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendLcarsMessage(); }
}

async function sendLcarsMessage() {
  const input = document.getElementById('lcars-input');
  const text = input.value.trim();
  if (!text) return;
  const btn = document.getElementById('lcars-send-btn');
  btn.disabled = true;
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('lcars-status-msg').textContent = 'SYS: PROCESSING...';
  addLcarsMessage('user', text);
  showLcarsTyping();
  await doChat(text, function(reply, err) {
    hideLcarsTyping();
    if (err) {
      addLcarsMessage('ai', 'ERROR: ' + err);
      document.getElementById('lcars-status-msg').textContent = 'SYS: ERROR';
    } else {
      addLcarsMessage('ai', reply);
      document.getElementById('lcars-status-msg').textContent = 'SYS: STANDBY';
    }
    btn.disabled = false;
  });
}

// ══════════════════════════════════════════════
// SHARED LOGIC
// ══════════════════════════════════════════════
async function loadConversations() {
  try {
    const resp = await fetch('/api/conversations');
    const convs = await resp.json();
    renderAppleConversations(convs);
    renderLcarsConversations(convs);
  } catch(e) {}
}

async function newConversation() {
  try {
    const resp = await fetch('/api/conversations', { method: 'POST' });
    const conv = await resp.json();
    currentConvId = conv.id;
    clearAppleChat();
    clearLcarsChat();
    await loadConversations();
  } catch(e) { showToast('Failed: ' + e.message); }
}

async function switchConversation(cid) {
  currentConvId = cid;
  try {
    const resp = await fetch('/api/conversations/' + cid);
    const conv = await resp.json();
    renderAppleMessages(conv.messages || []);
    renderLcarsMessages(conv.messages || []);
    await loadConversations();
  } catch(e) { showToast('Failed: ' + e.message); }
}

async function deleteConversation(cid) {
  if (!confirm('Delete this conversation?')) return;
  try {
    await fetch('/api/conversations/' + cid, { method: 'DELETE' });
    if (currentConvId === cid) {
      currentConvId = null;
      clearAppleChat();
      clearLcarsChat();
    }
    await loadConversations();
  } catch(e) { showToast('Failed: ' + e.message); }
}

async function doChat(text, cb) {
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, conversation_id: currentConvId || '' }),
    });
    const data = await resp.json();
    if (data.error) {
      cb(null, data.error);
    } else {
      currentConvId = data.conversation_id;
      cb(data.reply, null);
    }
    await loadConversations();
  } catch(e) {
    cb(null, e.message);
  }
}

// ══════════════════════════════════════════════
// SETTINGS
// ══════════════════════════════════════════════
async function openSettings() {
  document.getElementById('settingsOverlay').classList.add('show');
  document.getElementById('themeToggle').checked = isLcars();
  try {
    const resp = await fetch('/api/config');
    configCache = await resp.json();
    populateSettings(configCache);
  } catch(e) { showToast('Cannot load: ' + e.message); }
}

function closeSettings() {
  document.getElementById('settingsOverlay').classList.remove('show');
}

function switchSettingsTab(tab, evt) {
  document.querySelectorAll('.settings-tab').forEach(function(t){ t.classList.remove('active'); });
  document.querySelectorAll('.settings-tab-content').forEach(function(c){ c.classList.remove('active'); });
  evt.target.classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}

function populateSettings(cfg) {
  var tts = cfg.tts || {};
  document.getElementById('cfg-rate').value = tts.rate != null ? tts.rate : 3;
  document.getElementById('cfg-processing-rate').value = tts.processing_rate != null ? tts.processing_rate : 1;
  document.getElementById('cfg-volume').value = tts.volume != null ? tts.volume : 100;

  var wake = cfg.wake || {};
  document.getElementById('cfg-wake-threshold').value = wake.threshold != null ? wake.threshold : 0.15;
  document.getElementById('wake-threshold-val').textContent = wake.threshold != null ? wake.threshold : 0.15;
  document.getElementById('cfg-stop-threshold').value = wake.stop_energy_threshold != null ? wake.stop_energy_threshold : 1200;
  document.getElementById('stop-threshold-val').textContent = wake.stop_energy_threshold != null ? wake.stop_energy_threshold : 1200;

  fetch('/api/voices').then(function(r){ return r.json(); }).then(function(data){
    var voices = data.voices || [];
    var sel = document.getElementById('cfg-voice');
    sel.innerHTML = '';
    var selectedIdx = 0;
    voices.forEach(function(v, i){
      var opt = document.createElement('option');
      opt.value = v.id; opt.textContent = v.name;
      sel.appendChild(opt);
      if (tts.voice && v.id === tts.voice) selectedIdx = i;
    });
    sel.selectedIndex = selectedIdx;
  }).catch(function(){
    document.getElementById('cfg-voice').innerHTML = '<option value="zh-CN-XiaoyiNeural">晓伊 (中文女声)</option>';
  });

  var api = cfg.api || {};
  document.getElementById('cfg-api-base').value = api.base || '';
  document.getElementById('cfg-api-key').value = api.key || '';
  document.getElementById('cfg-api-model').value = api.model || '';
}

async function saveSettings() {
  var tts = {
    voice: document.getElementById('cfg-voice').value,
    rate: parseInt(document.getElementById('cfg-rate').value) || 3,
    processing_rate: parseInt(document.getElementById('cfg-processing-rate').value) || 1,
    volume: parseInt(document.getElementById('cfg-volume').value) || 100,
  };
  var api = {
    base: document.getElementById('cfg-api-base').value.trim(),
    key: document.getElementById('cfg-api-key').value.trim(),
    model: document.getElementById('cfg-api-model').value.trim(),
  };
  var wake = {
    threshold: parseFloat(document.getElementById('cfg-wake-threshold').value) || 0.15,
    stop_energy_threshold: parseInt(document.getElementById('cfg-stop-threshold').value) || 1200,
  };
  try {
    var resp = await fetch('/api/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tts: tts, api: api, wake: wake }),
    });
    var data = await resp.json();
    if (data.ok) { showToast('Saved. Restart voice assistant to apply.'); closeSettings(); }
    else { showToast('Save failed: ' + (data.error || '')); }
  } catch(e) { showToast('Save failed: ' + e.message); }
}

// ══════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════
function refreshAllUI() {
  // Re-render both layouts to sync active conversation highlighting
  loadConversations().then(function() {
    if (currentConvId) {
      fetch('/api/conversations/' + currentConvId).then(function(r){ return r.json(); }).then(function(conv){
        renderAppleMessages(conv.messages || []);
        renderLcarsMessages(conv.messages || []);
      }).catch(function(){});
    }
  });
}

loadConversations();

// Auto-resize textareas
document.getElementById('apple-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});
document.getElementById('lcars-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 160) + 'px';
});
