"""The reviewer's page: pending cases with the proposal, the evidence and three actions."""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>first-reply review</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5d6675; --line:#dfe3ea;
          --accent:#2a78d6; --ok:#1baf7a; --warn:#eda100; --bad:#d9534f; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14171c; --card:#1d2128; --ink:#e7eaf0; --muted:#9aa3b2; --line:#2c323c;
            --accent:#3987e5; --ok:#199e70; --warn:#c98500; --bad:#e06460; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
  header { padding:16px 24px; border-bottom:1px solid var(--line); display:flex; gap:16px;
           align-items:center; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; }
  header .muted { color:var(--muted); }
  main { max-width:1100px; margin:0 auto; padding:16px; display:grid; gap:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
  .row { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
  @media (max-width: 760px) { .row { grid-template-columns: 1fr; } }
  h2 { font-size:16px; margin:0 0 4px; } h3 { font-size:13px; text-transform:uppercase;
       letter-spacing:.04em; color:var(--muted); margin:14px 0 6px; }
  pre { white-space:pre-wrap; margin:0; font:inherit; }
  .chips span { display:inline-block; padding:1px 8px; margin:0 6px 6px 0; border-radius:999px;
                border:1px solid var(--line); font-size:13px; }
  .ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); }
  .evidence li { margin-bottom:8px; color:var(--muted); font-size:13px; }
  textarea { width:100%; min-height:140px; font:inherit; padding:8px; border-radius:8px;
             border:1px solid var(--line); background:var(--bg); color:var(--ink); }
  input { font:inherit; padding:6px 8px; border-radius:8px; border:1px solid var(--line);
          background:var(--bg); color:var(--ink); }
  button { font:inherit; padding:6px 14px; border-radius:8px; border:1px solid var(--line);
           background:var(--card); color:var(--ink); cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; align-items:center; }
  .empty { text-align:center; color:var(--muted); padding:40px; }
</style>
</head>
<body>
<header>
  <h1>Review queue</h1>
  <span class="muted" id="count"></span>
  <label class="muted">Reviewer <input id="reviewer" value="reviewer" size="12"></label>
  <button onclick="load()">Refresh</button>
</header>
<main id="list"></main>
<script>
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function load() {
  const cases = await (await fetch("cases?status=pending")).json();
  document.getElementById("count").textContent = cases.length + " pending";
  const list = document.getElementById("list");
  if (!cases.length) { list.innerHTML = '<div class="card empty">Nothing to review.</div>'; return; }
  const full = await Promise.all(cases.map(c => fetch("cases/" + c.id).then(r => r.json())));
  list.innerHTML = full.map(render).join("");
}
function render(c) {
  const p = c.proposal, r = p.routing.llm, v = p.routing.similar_tickets_vote, g = p.gate, d = p.draft;
  const conf = (v.queue.confidence * 100).toFixed(0);
  const route = g.auto_route ? '<span class="ok">routed automatically</span>'
                             : '<span class="warn">routing needs a look</span>';
  const answer = d.answerable ? "" : '<p class="bad">The knowledge base does not answer this. Write the reply or forward it.</p>';
  return `<section class="card" id="c-${c.id}">
    <h2>${esc(c.subject) || "(no subject)"}</h2>
    <div class="muted">${esc(c.sender)} · ${p.language} · case ${c.id}</div>
    <div class="row">
      <div>
        <h3>Email (personal data masked)</h3><pre>${esc(c.body_masked)}</pre>
        <h3>Routing</h3>
        <div class="chips"><span>${esc(r.queue)}</span><span>${esc(r.priority)}</span><span>${esc(r.type)}</span>
          ${(r.tags || []).map(t => `<span>${esc(t)}</span>`).join("")}</div>
        <div>${route} · similar tickets say <b>${esc(v.queue.label)}</b> (${conf}%)</div>
        ${g.reasons.length ? `<div class="muted">${g.reasons.map(esc).join("; ")}</div>` : ""}
      </div>
      <div>
        <h3>Draft reply</h3>${answer}
        <textarea id="reply-${c.id}">${esc(d.reply)}</textarea>
        <h3>Evidence</h3>
        <ol class="evidence">${p.evidence.map(e => `<li><b>[${e.n}]</b> ${esc(e.section)} — ${esc(e.snippet.slice(0, 220))}…</li>`).join("")}</ol>
      </div>
    </div>
    <div class="actions">
      <button class="primary" onclick="decide('${c.id}','approve')">Approve and send</button>
      <button onclick="decide('${c.id}','edit')">Send my edit</button>
      <button onclick="decide('${c.id}','reject')">Reject</button>
      <span class="muted" id="msg-${c.id}"></span>
    </div>
  </section>`;
}
async function decide(id, action) {
  const reviewer = document.getElementById("reviewer").value || "reviewer";
  const reply = document.getElementById("reply-" + id).value;
  const res = await fetch(`cases/${id}/decision`, {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({reviewer, action, reply: action === "edit" ? reply : null})});
  const msg = document.getElementById("msg-" + id);
  if (res.ok) { msg.textContent = action + "d"; setTimeout(load, 600); }
  else { msg.textContent = "error: " + (await res.text()); }
}
load();
</script>
</body>
</html>
"""
