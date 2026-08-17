"""Self-contained analytics dashboard, served by the backend at /analytics/dashboard.

A single HTML page with embedded CSS/JS (no build step, no third-party assets):
it fetches /analytics/summary on load and every 30s and renders KPI cards plus
top-query/click tables. Same-origin, so no CSP or CORS concerns.
"""

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASK VCCircle · Analytics</title>
<style>
  :root {
    --navy: #07162b;
    --navy2: #050e1d;
    --blue: #0875ff;
    --orange: #f83600;
    --ink: #222;
    --meta: #484848;
    --line: #e3e3e3;
    --paper: #fff;
    --bg: #f4f5f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--ink);
    min-height: 100vh;
  }
  header.topbar {
    background: linear-gradient(90deg, var(--navy) 0%, var(--navy2) 100%);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  header.topbar .brand { display: flex; align-items: center; gap: 10px; color: #fff; font-weight: 700; }
  header.topbar .brand .dot { color: var(--orange); }
  header.topbar .sub { color: rgba(255,255,255,.65); font-size: 12px; font-weight: 500; letter-spacing: .04em; }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 28px 24px 64px; }
  .updated { color: var(--meta); font-size: 12px; margin-bottom: 16px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }
  .card {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
  }
  .card .label { color: var(--meta); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
  .card .value { font-size: 28px; font-weight: 700; margin-top: 6px; }
  .card .value small { font-size: 14px; color: var(--meta); font-weight: 500; }
  .card .hint { font-size: 11px; color: var(--meta); margin-top: 4px; }
  .card .value.ok { color: var(--blue); }
  .card .value.warn { color: var(--orange); }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
  .panel { background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
  .panel h2 { font-size: 15px; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--meta); font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
       padding: 6px 8px; border-bottom: 2px solid var(--line); }
  td { padding: 8px; border-bottom: 1px solid var(--line); }
  tr:last-child td { border-bottom: none; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .bar { display: inline-block; height: 8px; background: var(--blue); border-radius: 4px; vertical-align: middle; }
  .empty { color: var(--meta); font-size: 13px; padding: 12px 0; }
  .err { background: #fdecea; border: 1px solid #f5c6c0; color: #a02020; border-radius: 8px; padding: 14px 16px; }
</style>
</head>
<body>
  <header class="topbar">
    <div class="brand">VCCircle <span class="dot">·</span> ASK — Analytics</div>
    <div class="sub">Search quality · self-hosted · no cookies</div>
  </header>
  <div class="wrap">
    <div class="updated" id="updated">loading…</div>
    <div id="error" class="err" style="display:none"></div>
    <div class="cards" id="cards"></div>
    <div class="grid2">
      <div class="panel">
        <h2>Top queries</h2>
        <div id="top-queries"><div class="empty">loading…</div></div>
      </div>
      <div class="panel">
        <h2>Clicks &amp; ask</h2>
        <div id="clicks"><div class="empty">loading…</div></div>
      </div>
    </div>
  </div>

<script>
  async function load() {
    try {
      var res = await fetch('/analytics/summary', { headers: { 'Accept': 'application/json' } });
      if (!res.ok) throw new Error('summary returned HTTP ' + res.status);
      var d = await res.json();
      document.getElementById('error').style.display = 'none';
      document.getElementById('updated').textContent =
        'Updated ' + new Date().toLocaleTimeString() + ' · counters since the analytics DB was last cleared';
      renderCards(d);
      renderTopQueries(d);
      renderClicks(d);
    } catch (e) {
      document.getElementById('error').textContent = 'Analytics unavailable: ' + e.message;
      document.getElementById('error').style.display = 'block';
    }
  }

  function fmt(n) { return (n == null || isNaN(n)) ? '0' : Number(n).toLocaleString(); }

  function card(label, value, hint, cls) {
    return '<div class="card"><div class="label">' + label + '</div>' +
      '<div class="value ' + (cls || '') + '">' + value + '</div>' +
      (hint ? '<div class="hint">' + hint + '</div>' : '') + '</div>';
  }

  function pct(v) { return v == null ? '0%' : v + '%'; }

  function renderCards(d) {
    var html =
      card('Searches today', fmt(d.searches_today), 'all time: ' + fmt(d.searches_total)) +
      card('Zero-result rate', pct(d.zero_result_rate), 'queries that found nothing', d.zero_result_rate > 15 ? 'warn' : 'ok') +
      card('Weak-result rate', pct(d.weak_result_rate), 'below relevance threshold', d.weak_result_rate > 25 ? 'warn' : 'ok') +
      card('Avg latency', d.avg_latency_ms + ' <small>ms</small>', 'server round-trip') +
      card('Cache hit', pct(d.cache_hit_rate), 'of searches served from cache') +
      card('Filtered', pct(d.filtered_rate), 'searches with facet/date filters') +
      card('Asks', fmt(d.asks_total), 'LLM answer requests') +
      card('Clicks', fmt(d.clicks_total), 'results opened by users', d.clicks_total > 0 ? 'ok' : '');
    document.getElementById('cards').innerHTML = html;
  }

  function topTable(items) {
    if (!items || !items.length) return '<div class="empty">No data yet.</div>';
    var max = items[0][1] || 1;
    var rows = items.map(function (r) {
      var pct = Math.round(100 * r[1] / max);
      return '<tr><td>' + r[0] + '</td>' +
        '<td class="num">' + fmt(r[1]) + '</td>' +
        '<td width="34%"><span class="bar" style="width:' + pct + '%"></span></td></tr>';
    }).join('');
    return '<table><tr><th>Query</th><th class="num">Count</th><th></th></tr>' + rows + '</table>';
  }

  function renderTopQueries(d) { document.getElementById('top-queries').innerHTML = topTable(d.top_queries); }

  function renderClicks(d) {
    var pos = d.click_positions || {};
    var posRows = Object.keys(pos).slice(0, 10).map(function (k) {
      return '<tr><td>Position ' + k + '</td><td class="num">' + fmt(pos[k]) + '</td></tr>';
    }).join('');
    var html = '';
    if (d.clicks_total > 0) {
      html += '<h2>Clicks by position</h2>' + '<table><tr><th>Result slot</th><th class="num">Clicks</th></tr>' + posRows + '</table>' +
        '<h2 style="margin-top:18px">Most-clicked queries</h2>' + topTable(d.click_top_queries);
    } else {
      html = '<div class="empty">No clicks recorded yet. Click a result link to start tracking.</div>';
    }
    document.getElementById('clicks').innerHTML = html;
  }

  load();
  setInterval(load, 30000);
</script>
</body>
</html>
"""


def dashboard_html() -> str:
    """The dashboard page as an HTML string (self-contained, no external assets)."""
    return DASHBOARD_HTML
