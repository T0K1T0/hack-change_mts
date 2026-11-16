(() => {
  const $ = id => document.getElementById(id);
  const auth = $('auth-screen'), login = $('login-form'), loginError = $('auth-error');
  const form = $('cform'), input = $('cin'), send = $('csend'), log = $('clog');
  const list = $('chat-list'), tools = $('chat-tools'), error = $('chat-error'), summary = $('csum');
  const create = $('chat-new'), remove = $('chat-delete'), logout = $('chat-logout');
  let csrf = '', chatId = null, busy = false, opening = 0;
  $('chips').replaceChildren();

  async function api(url, options = {}) {
    let response;
    try {
      response = await fetch(url, { ...options, credentials: 'same-origin',
        headers: {'Content-Type': 'application/json', ...(csrf ? {'X-CSRF-Token': csrf} : {})} });
    } catch { throw new Error('Нет соединения с сервером. Проверьте сеть и повторите попытку.'); }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401 && url !== '/api/auth/login') showLogin();
      const message = response.status === 502 || response.status === 503
        ? 'Консультант сейчас недоступен. Ваш вопрос сохранён; попробуйте позже.'
        : (typeof data.detail === 'string' ? data.detail : `Ошибка сервера (${response.status})`);
      const failure = new Error(message);
      failure.status = response.status;
      throw failure;
    }
    return data;
  }

  function setError(message = '') { error.textContent = message; error.hidden = !message; }
  function showLogin() {
    csrf = ''; chatId = null; opening++;
    auth.hidden = false; form.hidden = true; tools.hidden = true;
    log.replaceChildren(); list.replaceChildren(); summary.hidden = true; setError();
  }
  function showApp() {
    auth.hidden = true; form.hidden = false; tools.hidden = false;
    $('end').scrollIntoView({behavior: 'smooth', block: 'start'});
  }
  function line(role, content, waiting = false) {
    const row = document.createElement('div'); row.className = 'ln ' + (role === 'user' ? 'me' : 'bot');
    const name = document.createElement('b'); name.textContent = role === 'user' ? 'вы' : 'консультант';
    const body = document.createElement('span'); body.textContent = content;
    if (waiting) body.classList.add('chat-wait');
    row.append(name, body); log.appendChild(row); log.scrollTop = log.scrollHeight;
    return row;
  }
  function showLead(lead) {
    summary.replaceChildren(); summary.hidden = !lead || lead.requested !== true;
    if (summary.hidden) return;
    const title = document.createElement('b'); title.textContent = 'Заявка в этом диалоге'; summary.appendChild(title);
    for (const [label, value] of [['Имя', lead.name], ['Компания', lead.company], ['Телефон', lead.phone], ['Описание', lead.summary]]) {
      if (!value) continue;
      const row = document.createElement('div'); row.className = 'sr set';
      const key = document.createElement('u'); key.textContent = label;
      const text = document.createElement('span'); text.textContent = value;
      row.append(key, text); summary.appendChild(row);
    }
    const state = document.createElement('p');
    state.textContent = lead.status === 'ready'
      ? 'Заявка сформирована локально. Отправка менеджеру не выполнялась.'
      : 'Черновик заявки. Отправка менеджеру не выполнялась.';
    summary.appendChild(state);
  }
  function setBusy(value) {
    busy = value; send.disabled = value; input.disabled = value;
    create.disabled = value; remove.disabled = value || !chatId; logout.disabled = value;
    list.querySelectorAll('button').forEach(button => { button.disabled = value; });
  }
  function renderList(chats) {
    list.replaceChildren();
    chats.forEach(chat => {
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = chat.title; button.title = chat.title;
      button.dataset.id = String(chat.id);
      button.classList.toggle('active', chat.id === chatId);
      button.disabled = busy;
      button.addEventListener('click', () => openChat(chat.id).catch(e => setError(e.message)));
      list.appendChild(button);
    });
  }
  async function loadChats(preferred = chatId) {
    const chats = await api('/api/chats'); renderList(chats);
    const selected = chats.some(chat => chat.id === preferred) ? preferred : chats[0]?.id;
    if (selected) await openChat(selected);
    else { chatId = null; log.replaceChildren(); line('assistant', 'Создайте диалог, чтобы задать вопрос консультанту.'); showLead(null); setBusy(false); }
  }
  async function openChat(id) {
    if (busy) return;
    const requestId = ++opening;
    const chat = await api(`/api/chats/${id}`);
    if (requestId !== opening) return;
    chatId = id; log.replaceChildren();
    chat.messages.forEach(message => line(message.role, message.content));
    if (!chat.messages.length) line('assistant', 'Здравствуйте! Расскажите о вашей задаче.');
    showLead(null); setError(); setBusy(false);
    list.querySelectorAll('button').forEach(button => button.classList.toggle('active', button.dataset.id === String(id)));
  }
  async function createChat(title = 'Новый диалог') {
    const chat = await api('/api/chats', {method: 'POST', body: JSON.stringify({title})});
    chatId = chat.id; await loadChats(chat.id); input.focus({preventScroll: true});
  }

  login.addEventListener('submit', async event => {
    event.preventDefault(); loginError.textContent = '';
    const button = $('login-submit'); button.disabled = true;
    try {
      const data = await api('/api/auth/login', {method: 'POST', body: JSON.stringify({
        username: $('login-name').value.trim(), password: $('login-password').value})});
      csrf = data.csrf_token; $('login-password').value = '';
      showApp(); await loadChats();
    } catch (e) { loginError.textContent = e.message; }
    finally { button.disabled = false; }
  });
  create.addEventListener('click', () => createChat().catch(e => setError(e.message)));
  remove.addEventListener('click', async () => {
    if (!chatId || busy) return;
    try { await api(`/api/chats/${chatId}`, {method: 'DELETE'}); chatId = null; await loadChats(); }
    catch (e) { setError(e.message); }
  });
  logout.addEventListener('click', async () => {
    try { await api('/api/auth/logout', {method: 'POST'}); showLogin(); }
    catch (e) { setError(e.message); }
  });
  form.addEventListener('submit', async event => {
    event.preventDefault(); const content = input.value.trim();
    if (!content || busy || !csrf) return;
    const previous = input.value; setError(); setBusy(true);
    try {
      if (!chatId) {
        const chat = await api('/api/chats', {method: 'POST', body: JSON.stringify({title: content.slice(0, 80)})});
        chatId = chat.id;
        renderList(await api('/api/chats'));
      }
      const sent = line('user', content); input.value = '';
      const pending = line('assistant', 'Консультант готовит ответ', true);
      try {
        const result = await api(`/api/chats/${chatId}/messages`, {method: 'POST', body: JSON.stringify({content})});
        pending.lastChild.textContent = result.answer;
        pending.lastChild.classList.remove('chat-wait');
        showLead(result.lead);
      } catch (e) {
        pending.remove();
        if (e.status !== 502 && e.status !== 503) { sent.remove(); input.value = previous; }
        setError(e.message);
      }
    } catch (e) { input.value = previous; setError(e.message); }
    finally { setBusy(false); input.focus({preventScroll: true}); }
  });
  api('/api/auth/me').then(async data => { csrf = data.csrf_token; showApp(); await loadChats(); })
    .catch(e => { showLogin(); if (!/Требуется вход|Сессия истекла/.test(e.message)) loginError.textContent = e.message; });
})();
