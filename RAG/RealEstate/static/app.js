const form = document.getElementById('chat-form');
const chat = document.getElementById('chat');
const questionInput = document.getElementById('question');

function appendMessage(text, who='bot'){
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
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
    } else {
      appendMessage('Error: ' + (data.error || 'unknown'), 'bot');
    }
  } catch (err) {
    appendMessage('Network error: ' + err.message, 'bot');
  }
});
