const form = document.getElementById('chat-form');
const chat = document.getElementById('chat');
const questionInput = document.getElementById('question');

function _escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function _renderAnswer(text){
  if(!text) return '';
  // Escape HTML to avoid injection, then convert double newlines to paragraphs
  let safe = _escapeHtml(text);
  // Normalize CRLF
  safe = safe.replace(/\r\n/g, '\n');
  // Split on double newlines into paragraphs
  const parts = safe.split(/\n\n+/g).map(p => p.trim()).filter(Boolean);
  const html = parts.map(p => p.replace(/\n/g, '<br>')).map(p => `<p>${p}</p>`).join('');
  return html || _escapeHtml(text);
}

function appendMessage(text, who='bot'){
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if(who === 'bot'){
    bubble.innerHTML = _renderAnswer(text);
  } else {
    bubble.textContent = text;
  }
  div.appendChild(bubble);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const q = questionInput.value.trim();
  if(!q) return;
  appendMessage(q, 'user');
  questionInput.value = '';
  appendMessage('Thinking...', 'bot');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    // remove the 'Thinking...' node
    const thinking = document.querySelector('.msg.bot:last-child');
    if(thinking && thinking.textContent === 'Thinking...') thinking.remove();

    if(data.success){
      appendMessage(data.answer, 'bot');
      // render sources if present
      const sourcesPanel = document.getElementById('sources-panel');
      const sourcesContent = document.getElementById('sources-content');
      const toggleBtn = document.getElementById('toggle-sources');
      if(data.sources){
        sourcesContent.innerHTML = '';
        // split sources by newlines and create items
        data.sources.split('\n').map(s => s.trim()).filter(Boolean).forEach(s => {
          const el = document.createElement('div');
          el.className = 'source-item';
          el.textContent = s;
          sourcesContent.appendChild(el);
        });
        sourcesPanel.style.display = '';
        // show collapsed by default
        sourcesContent.style.display = 'none';
        toggleBtn.textContent = 'Show sources';
      } else {
        sourcesPanel.style.display = 'none';
      }
    } else {
      appendMessage('Error: ' + (data.error || 'unknown'), 'bot');
    }
  } catch (err) {
    appendMessage('Network error: ' + err.message, 'bot');
  }
});

// Toggle behavior for sources panel
document.addEventListener('click', (ev) => {
  if(ev.target && ev.target.id === 'toggle-sources'){
    const sourcesContent = document.getElementById('sources-content');
    const toggleBtn = document.getElementById('toggle-sources');
    if(sourcesContent.style.display === 'none'){
      sourcesContent.style.display = '';
      toggleBtn.textContent = 'Hide sources';
    } else {
      sourcesContent.style.display = 'none';
      toggleBtn.textContent = 'Show sources';
    }
  }
});
