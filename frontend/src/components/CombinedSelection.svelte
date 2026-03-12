<script>
  import { createEventDispatcher } from 'svelte'
  // CombinedSelection: show positive (top) and negative (bottom) previews, with apply/rename/delete and collapsible sections
  export let combined // { id, name, posIds: string[], negIds: string[] }
  export let idToUrl = new Map() // Map id -> url
  const dispatch = createEventDispatcher()

  let expandedPos = true
  let expandedNeg = true

  function getUrl(id) { return idToUrl instanceof Map ? (idToUrl.get(id) || '') : (idToUrl?.[id] || '') }
  function onApply() { dispatch('apply', { id: combined?.id, posIds: combined?.posIds || [], negIds: combined?.negIds || [] }) }
  function onRename() {
    const next = prompt('Rename combined selection', combined?.name || 'Combined selection')
    if (next && next.trim()) dispatch('rename', { id: combined?.id, name: next.trim() })
  }
  function onDelete() { dispatch('delete', { id: combined?.id }) }

  function toLargeUrl(u) {
    if (!u || typeof u !== 'string') return u
    // Prefer larger thumbnails if available
    try {
      if (u.includes('/thumb/')) {
        return u.replace(/\/thumb\/(\d+)\//, '/thumb/800/')
      }
    } catch (_) {}
    return u
  }
  function openPopupView() {
    const name = combined?.name || 'Combined selection'
    const pos = (combined?.posIds || []).map(id => ({ id, url: toLargeUrl(getUrl(id)) })).filter(it => it.url)
    const neg = (combined?.negIds || []).map(id => ({ id, url: toLargeUrl(getUrl(id)) })).filter(it => it.url)
    const total = pos.length + neg.length
    const win = window.open('', '_blank')
    if (!win) return
    const css = `
      html, body { margin:0; padding:0; font-family: Inter, sans-serif; background:#fff; color:#111; }
      .bar { position: sticky; top:0; z-index:10; background:#ffffff; border-bottom:1px solid #e5e7eb; padding:8px 12px; display:flex; align-items:center; gap:12px; }
      .title { font-weight:700; }
      .muted { color:#475569; font-size: 0.9rem; }
      .section { padding: 10px 12px; }
      .section h2 { margin: 6px 0 10px; font-size: 1rem; color:#334155; }
      .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap:10px; }
      .item { border:1px solid #e5e7eb; border-radius:10px; overflow:hidden; background:#fff; }
      .item img { display:block; width:100%; height:auto; }
      .cap { padding:6px 8px; font-size:.85rem; color:#475569; word-break: break-all; }
    `
    const section = (label, arr) => `
      <div class="section">
        <h2>${label} (${arr.length})</h2>
        <div class="grid">
          ${arr.map(it => `<div class="item"><img src="${it.url}" alt="${it.id}"><div class="cap">${it.id}</div></div>`).join('')}
        </div>
      </div>
    `
    const html = `<!doctype html>
      <html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
        <title>${name}</title>
        <style>${css}</style>
      </head>
      <body>
        <div class="bar"><div class="title">${name}</div><div class="muted">${total} images</div></div>
        ${section('Positive examples', pos)}
        ${section('Negative examples', neg)}
      </body></html>`
    win.document.open()
    win.document.write(html)
    win.document.close()
  }
</script>

<div class="tile">
  <div class="tile-header">
    <div class="truncate flex-1" title={`Pos: ${(combined?.posIds||[]).length} • Neg: ${(combined?.negIds||[]).length}`}>{combined?.name || 'Combined selection'}</div>
    <div class="shrink-0 inline-flex gap-1">
      <button class="btn btn-xs" on:click={onApply} title="Apply to labels">↻</button>
      <button class="btn btn-xs" on:click={openPopupView} title="Open large image viewer">⤢</button>
      <button class="btn btn-xs" on:click={onRename} title="Rename">✎</button>
      <button class="btn btn-xs btn-danger" on:click={onDelete} title="Delete">✕</button>
      <button class="btn btn-xs" on:click={() => dispatch('sendToSelections', { selection: { id: `sel:${Date.now()}`, name: combined?.name || 'Selection', posIds: combined?.posIds||[], negIds: combined?.negIds||[], active: true } })} title="Move to selections">⇄</button>
    </div>
  </div>

  <div class="tile-content">
  <!-- Top: Positives -->
  <div class="mb-2">
    <div class="flex items-center justify-between">
      <div class="text-sm text-gray-600">Positive examples ({(combined?.posIds||[]).length})</div>
      <button class="text-sm px-1 py-0.5 border rounded" on:click={() => (expandedPos = !expandedPos)}>{expandedPos ? 'Hide' : 'Show'}</button>
    </div>
    {#if expandedPos}
      <div class="mt-1 flex flex-wrap gap-1" style="max-height: 180px; overflow:auto">
        {#each (combined?.posIds || []).slice(0,200) as id}
          {#if getUrl(id)}
            <img src={getUrl(id)} alt={id} class="w-12 h-12 object-cover rounded border" loading="lazy" />
          {/if}
        {/each}
      </div>
    {/if}
  </div>

  <!-- Bottom: Negatives -->
  <div>
    <div class="flex items-center justify-between">
      <div class="text-sm text-gray-600">Negative examples ({(combined?.negIds||[]).length})</div>
      <button class="text-sm px-1 py-0.5 border rounded" on:click={() => (expandedNeg = !expandedNeg)}>{expandedNeg ? 'Hide' : 'Show'}</button>
    </div>
    {#if expandedNeg}
      <div class="mt-1 flex flex-wrap gap-1" style="max-height: 180px; overflow:auto">
        {#each (combined?.negIds || []).slice(0,200) as id}
          {#if getUrl(id)}
            <img src={getUrl(id)} alt={id} class="w-12 h-12 object-cover rounded border" loading="lazy" />
          {/if}
        {/each}
      </div>
    {/if}
  </div>
  </div>
</div>

<style>
</style>
