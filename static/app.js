/* TruthLens — frontend.
 * All dynamic content is inserted with textContent / DOM APIs (never innerHTML),
 * so text coming from articles or the LLM cannot inject markup. */
(function () {
  'use strict';

  const HISTORY_KEY = 'truthlens.history.v2';
  const HISTORY_MAX = 10;
  const MAX_TEXT = 5000;

  const VERDICTS = {
    LIKELY_FAKE: { label: 'Likely fake', tone: 'fake',
      sub: 'Strong warning signs. Verify with trusted sources before believing or sharing.' },
    UNCERTAIN: { label: 'Uncertain', tone: 'warn',
      sub: 'Mixed signals. Check the claims below against primary sources.' },
    LIKELY_REAL: { label: 'Likely credible', tone: 'real',
      sub: 'No strong warning signs found. That does not prove it is true.' },
    INSUFFICIENT_SIGNAL: { label: 'Not enough signal', tone: 'none',
      sub: 'TruthLens could not assess this text. Try the full headline or add an AI key on the server.' },
  };

  const EXAMPLES = [
    'Pope Francis shocks world, endorses Donald Trump for president',
    'Federal Reserve raises interest rates by a quarter point amid inflation concerns',
    'BREAKING: Celebrity couple secretly split, insiders reveal shocking truth',
  ];

  // ── Tiny DOM helper ─────────────────────────────────────────
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  const $ = (id) => document.getElementById(id);

  // ── Storage (best effort — may be blocked) ──────────────────
  function loadHistory() {
    try {
      const raw = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
      return Array.isArray(raw) ? raw.slice(0, HISTORY_MAX) : [];
    } catch (_) { return []; }
  }
  function saveHistory(items) {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); } catch (_) { /* ignore */ }
  }
  let history = loadHistory();

  // ── Tabs ────────────────────────────────────────────────────
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  function selectTab(name, focus) {
    for (const tab of tabs) {
      const on = tab.dataset.tab === name;
      tab.setAttribute('aria-selected', String(on));
      tab.tabIndex = on ? 0 : -1;
      $(tab.getAttribute('aria-controls')).hidden = !on;
      if (on && focus) tab.focus();
    }
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => selectTab(tab.dataset.tab));
    tab.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      selectTab(next.dataset.tab, true);
    });
  });

  // ── Requests ────────────────────────────────────────────────
  function setLoading(form, on) {
    const btn = form.querySelector('button[type="submit"]');
    btn.disabled = on;
    btn.classList.toggle('is-loading', on);
    btn.querySelector('.btn-label').textContent = on ? 'Analysing…' : 'Analyse';
    $('result').setAttribute('aria-busy', String(on));
  }

  async function post(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60000);
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON */ }
      if (!res.ok) {
        throw new Error((data && data.error) || `Request failed (HTTP ${res.status}).`);
      }
      return data;
    } catch (e) {
      if (e.name === 'AbortError') throw new Error('The request took too long. Please try again.');
      if (e instanceof TypeError) throw new Error('Network error. Is the server running?');
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  async function submit(form, path, body) {
    setLoading(form, true);
    $('result').replaceChildren();
    try {
      const data = await post(path, body);
      renderResult(data);
      addHistory(data, body);
    } catch (e) {
      renderError(e.message);
    } finally {
      setLoading(form, false);
    }
  }

  $('panel-url').addEventListener('submit', (e) => {
    e.preventDefault();
    const input = $('url-input');
    const url = input.value.trim();
    if (!url) { input.focus(); return renderError('Please paste a news article URL.'); }
    submit(e.currentTarget, '/api/verify-url', { url });
  });

  $('panel-text').addEventListener('submit', (e) => {
    e.preventDefault();
    const input = $('text-input');
    const text = input.value.trim();
    if (text.length < 5) { input.focus(); return renderError('Please enter a headline or some text (at least 5 characters).'); }
    if (text.length > MAX_TEXT) return renderError(`Please keep the text under ${MAX_TEXT.toLocaleString()} characters.`);
    submit(e.currentTarget, '/api/verify-text', { text });
  });

  $('text-input').addEventListener('input', updateCount);
  $('text-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) $('panel-text').requestSubmit();
  });
  function updateCount() {
    const n = $('text-input').value.length;
    const el = $('char-count');
    el.textContent = n ? `${n.toLocaleString()} / ${MAX_TEXT.toLocaleString()}` : '';
    el.classList.toggle('over', n > MAX_TEXT);
  }

  const examplesEl = $('examples');
  for (const ex of EXAMPLES) {
    const btn = h('button', { type: 'button', class: 'chip-btn', title: ex, text: ex });
    btn.addEventListener('click', () => {
      $('text-input').value = ex;
      updateCount();
      $('panel-text').requestSubmit();
    });
    examplesEl.append(btn);
  }

  // ── Rendering ───────────────────────────────────────────────
  function renderError(message) {
    $('result').replaceChildren(
      h('div', { class: 'error-card result', role: 'alert' },
        h('span', { 'aria-hidden': 'true', text: '⚠' }), h('span', { text: message })));
  }

  function verdictMeta(v) { return VERDICTS[v] || VERDICTS.INSUFFICIENT_SIGNAL; }

  function mlCard(ml) {
    const card = h('article', { class: 'signal' });
    card.append(h('h3', {}, 'Wording signal', h('small', { text: 'ML · headline classifier' })));
    if (ml.risk_score === null || ml.risk_score === undefined) {
      card.append(h('p', { class: 'muted', text: 'None of these words appear in the model\'s vocabulary, so it can\'t score this text.' }));
      return card;
    }
    const tone = { HIGH: 'fake', MEDIUM: 'warn', LOW: 'real' }[ml.risk_level] || 'none';
    const dot = h('span', { class: 'meter-dot' });
    card.append(
      h('div', { class: 'big-num' }, `${ml.risk_pct}%`, h('small', { text: 'fake-like' })),
      h('div', { class: 'meter', role: 'img', 'aria-label': `Fake-likeness ${ml.risk_pct} percent` }, dot),
      h('div', { class: 'meter-scale' }, h('span', { text: 'real-like' }), h('span', { text: 'uncertain' }), h('span', { text: 'fake-like' })),
      h('dl', { class: 'kv' },
        h('dt', { text: 'Band' }), h('dd', { text: `${ml.risk_level} risk` }),
        h('dt', { text: 'Coverage' }), h('dd', { text: `${Math.round((ml.coverage || 0) * 100)}% of words known${ml.low_signal ? ' (weak evidence)' : ''}` })),
    );
    // Style via the CSSOM — inline style attributes are blocked by the page's CSP.
    card.querySelector('.big-num').style.color = `var(--${tone})`;
    dot.style.left = `${Math.min(100, Math.max(0, ml.risk_pct))}%`;

    const exp = ml.explanation || {};
    if ((exp.toward_fake || []).length || (exp.toward_real || []).length) {
      const terms = h('div', { class: 'terms' }, h('div', { class: 'subhead', text: 'Words that moved the score' }));
      if ((exp.toward_fake || []).length) {
        terms.append(h('div', { class: 'terms-row', 'aria-label': 'toward fake' },
          exp.toward_fake.map((t) => h('span', { class: 'term fake', title: `+${t.weight}`, text: `↑ ${t.term}` }))));
      }
      if ((exp.toward_real || []).length) {
        terms.append(h('div', { class: 'terms-row', 'aria-label': 'toward real' },
          exp.toward_real.map((t) => h('span', { class: 'term real', title: `${t.weight}`, text: `↓ ${t.term}` }))));
      }
      card.append(terms);
    }
    return card;
  }

  function factLinks(claim) {
    const q = encodeURIComponent(claim);
    const link = (href, label) => h('a', { href, target: '_blank', rel: 'noopener noreferrer', text: label });
    return h('div', { class: 'claim-links' },
      link(`https://www.snopes.com/search/?q=${q}`, 'Snopes →'),
      link(`https://www.factcheck.org/?s=${q}`, 'FactCheck.org →'),
      link(`https://www.google.com/search?q=${encodeURIComponent('fact check ' + claim)}`, 'Search →'));
  }

  function llmCard(llm) {
    const card = h('article', { class: 'signal' });
    card.append(h('h3', {}, 'AI assessment', h('small', { text: llm.model ? `LLM · ${llm.model}` : 'LLM' })));
    if (llm.status === 'disabled') {
      card.append(h('p', { class: 'muted', text: 'AI analysis is not enabled on this server. The administrator can enable it by setting GROQ_API_KEY.' }));
      return card;
    }
    if (llm.status !== 'ok') {
      card.append(h('p', { class: 'muted', text: llm.message || 'AI analysis is unavailable right now.' }));
      return card;
    }
    const tone = { 'FAKE': 'fake', 'LIKELY FAKE': 'fake', 'UNCERTAIN': 'warn', 'LIKELY REAL': 'real', 'REAL': 'real' }[llm.verdict] || 'none';
    const num = h('div', { class: 'big-num' }, `${llm.credibility_score}`, h('small', { text: '/ 100 credibility' }));
    num.style.color = `var(--${tone})`;
    card.append(num,
      h('dl', { class: 'kv' },
        h('dt', { text: 'Verdict' }), h('dd', { text: llm.verdict }),
        h('dt', { text: 'Confidence' }), h('dd', { text: `${llm.confidence}/100` })));
    if (llm.summary) card.append(h('p', { class: 'llm-summary', text: llm.summary }));

    const list = (title, items, cls) => (items && items.length)
      ? h('div', { class: 'section' }, h('div', { class: 'subhead', text: title }),
          h('ul', { class: `flag-list ${cls}` }, items.map((x) => h('li', { text: x }))))
      : null;
    card.append(
      list('Red flags', llm.red_flags, 'red'),
      list('Credibility signals', llm.green_flags, 'green'),
    );
    if ((llm.detected_techniques || []).length) {
      card.append(h('div', { class: 'section' }, h('div', { class: 'subhead', text: 'Manipulation techniques' }),
        h('div', { class: 'terms-row' }, llm.detected_techniques.map((t) => h('span', { class: 'term fake', text: t })))));
    }
    if (llm.reasoning) {
      card.append(h('details', { class: 'reasoning' }, h('summary', { text: 'Show AI reasoning' }), h('p', { text: llm.reasoning })));
    }
    return card;
  }

  function renderResult(data) {
    const meta = verdictMeta(data.verdict);
    const badges = h('div', { class: 'badges' },
      h('span', { class: 'badge accent', text: data.verdict_basis === 'llm+ml' ? 'AI + ML' : 'ML only' }),
      data.agreement === 'disagree' ? h('span', { class: 'badge warn', text: 'Signals disagree' }) : null);

    const verdict = h('div', { class: `verdict tone-${meta.tone}` },
      h('div', { class: 'verdict-top' },
        h('div', {}, h('h2', { class: 'verdict-label', text: meta.label }), h('p', { class: 'verdict-sub', text: meta.sub })),
        badges));
    if ((data.notes || []).length) {
      verdict.append(h('div', { class: 'notes' }, data.notes.map((n) => h('p', { class: 'note', text: n }))));
    }

    const signals = h('div', { class: 'signals' }, mlCard(data.ml || {}), llmCard(data.llm || { status: 'disabled' }));

    const claims = (data.llm && data.llm.status === 'ok' && (data.llm.fact_check_suggestions || []).length)
      ? h('div', { class: 'card claims' }, h('div', { class: 'subhead', text: 'Claims to verify yourself' }),
          data.llm.fact_check_suggestions.map((c, i) => h('div', { class: 'claim' }, h('p', { text: `${i + 1}. ${c}` }), factLinks(c))))
      : null;

    const rows = [];
    if (data.headline) rows.push(h('dt', { text: 'Headline' }), h('dd', { class: 'headline', text: data.headline }));
    if (data.source) rows.push(h('dt', { text: 'Source' }), h('dd', { text: data.source }));
    if (data.preview) rows.push(h('dt', { text: 'Excerpt' }), h('dd', { class: 'muted', text: data.preview }));
    if (data.url && /^https?:\/\//i.test(data.url)) {
      rows.push(h('dt', { text: 'Link' }), h('dd', {}, h('a', { href: data.url, target: '_blank', rel: 'noopener noreferrer', text: 'Open original ↗' })));
    }
    const article = rows.length ? h('div', { class: 'card article' }, h('dl', { class: 'kv' }, rows)) : null;

    const wrap = h('div', { class: 'result' }, verdict, signals, claims, article);
    $('result').replaceChildren(wrap);
    wrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ── History ─────────────────────────────────────────────────
  function addHistory(data, request) {
    const key = request.url || request.text;
    history = history.filter((x) => x.key !== key);
    history.unshift({ key, at: Date.now(), data });
    history = history.slice(0, HISTORY_MAX);
    saveHistory(history);
    renderHistory();
  }

  function renderHistory() {
    const section = $('history');
    const list = $('history-list');
    section.hidden = history.length === 0;
    list.replaceChildren(...history.map((item) => {
      const meta = verdictMeta(item.data.verdict);
      const dot = h('span', { class: 'dot', 'aria-hidden': 'true' });
      dot.style.background = `var(--${meta.tone})`;
      const btn = h('button', { type: 'button', class: 'history-item' },
        dot,
        h('span', { class: 'history-headline', text: item.data.headline || item.key }),
        h('span', { class: 'history-meta', text: meta.label }));
      btn.addEventListener('click', () => renderResult(item.data));
      return h('li', {}, btn);
    }));
  }

  $('clear-history').addEventListener('click', () => {
    history = [];
    saveHistory(history);
    renderHistory();
  });

  // ── Model info ──────────────────────────────────────────────
  async function loadModelInfo() {
    try {
      const res = await fetch('/api/model-info');
      if (!res.ok) return;
      const info = await res.json();
      if (info.is_demo) $('demo-banner').hidden = false;
      const ds = info.dataset || {};
      if (ds.name) $('about-dataset').textContent = `${ds.name} (${(ds.rows || 0).toLocaleString()} headlines)`;
      const test = (info.metrics && info.metrics.test) || {};
      const bands = (info.metrics && info.metrics.bands) || {};
      const stat = (value, label) => h('div', { class: 'stat' }, h('b', { text: value }), h('span', { text: label }));
      const pct = (x) => (typeof x === 'number' ? `${Math.round(x * 100)}%` : 'n/a');
      const stats = [];
      if (test.roc_auc !== undefined) stats.push(stat(test.roc_auc.toFixed(2), 'ROC-AUC (held-out)'));
      if (test.accuracy !== undefined) stats.push(stat(pct(test.accuracy), 'Accuracy'));
      if (bands.likely_fake) stats.push(stat(pct(bands.likely_fake.actually_fake), '"Likely fake" correct'));
      if (bands.likely_real) stats.push(stat(pct(1 - bands.likely_real.actually_fake), '"Likely real" correct'));
      $('about-stats').replaceChildren(...stats);
      $('footer-model').textContent = `${info.model_name || 'ML model'} · AI layer ${info.llm_enabled ? 'on' : 'off'}`;
    } catch (_) { /* informational only */ }
  }

  renderHistory();
  updateCount();
  loadModelInfo();
})();
