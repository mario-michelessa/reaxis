<script>
  // AxesMinimap: like HoverGridMinimap, but x/y come from selected 1D axes.
  // - Drag axes from AxesPanel into X/Y dropzones to change current axes
  // - Uses local packing inside selection window for image grid placement

  import { createEventDispatcher, onMount, onDestroy } from 'svelte'
  import LassoSelector from './LassoSelector.svelte'
  import Callout from './Callout.svelte'
  export let items = [] // [{ id, url, x, y, gx, gy }]
  export let axes = [] // [{ id, name, coords: Record<string, number> }]
  export let width = 700
  export let height = 700
  export let viewFrac = 0.15 // viewport square fraction (initial)
  export let duration = 400 // ms
  export let minImagePx = 20
  export let posUpdateMs = 450
  const dispatch = createEventDispatcher()

  // Visual margin for display (map [0,1] -> [m, 1-m])
  export let displayMargin = 0.1
  function toVis(v) {
    const m = Math.max(0, Math.min(0.49, Number(displayMargin || 0)))
    const nv = Math.max(0, Math.min(1, Number(v || 0)))
    return m + nv * (1 - 2 * m)
  }

  // External labels for outline rendering and axis creation
  export let labels = new Map()
  function labelOf(id) {
    try {
      if (!labels) return undefined
      if (labels instanceof Map) return labels.get(id)
      return labels[id]
    } catch (_) { return undefined }
  }

  // Counts for labeled items
  $: posCount = (() => {
    let c = 0
    if (!labels) return 0
    try {
      if (labels instanceof Map) { labels.forEach((v) => { if (v === 'good') c++ }) }
      else { for (const v of Object.values(labels)) if (v === 'good') c++ }
    } catch (_) {}
    return c
  })()
  $: negCount = (() => {
    let c = 0
    if (!labels) return 0
    try {
      if (labels instanceof Map) { labels.forEach((v) => { if (v === 'bad') c++ }) }
      else { for (const v of Object.values(labels)) if (v === 'bad') c++ }
    } catch (_) {}
    return c
  })()

  // Lasso selection & labeling
  let lassoEnabled = false
  let lassoMode = 'pos' // 'pos' | 'neg' controls lasso color
  let lastSelectedIds = []
  let lassoRef
  function onLassoSelect(e) {
    const ids = Array.isArray(e.detail && e.detail.ids) ? e.detail.ids : []
    lastSelectedIds = ids
    if (!ids || ids.length === 0) {
      try { lassoRef && lassoRef.reset && lassoRef.reset() } catch(_) {}
      return
    }
    // Build updates according to lassoMode and current labels
    const updates = []
    for (const id of ids) {
      const cur = labelOf(id)
      if (lassoMode === 'pos') {
        if (cur === 'good') updates.push({ id, label: null }) // toggle off
        else updates.push({ id, label: 'good' }) // set or flip to good
      } else {
        if (cur === 'bad') updates.push({ id, label: null })
        else updates.push({ id, label: 'bad' })
      }
    }
    dispatch('label', { updates })
    // Clear selection and lasso path; keep lasso enabled for next selection
    lastSelectedIds = []
    try { lassoRef && lassoRef.reset && lassoRef.reset() } catch(_) {}
  }
  // Buttons now only change mode (color); application happens on release
  function setLassoMode(kind) { lassoMode = (kind === 'neg') ? 'neg' : 'pos' }
  function createAxisFromLabels() {
    // Build axis scores based on current labels
    const goodIds = new Set()
    const badIds = new Set()
    if (labels) {
      if (labels instanceof Map) {
        labels.forEach((v, k) => { if (v === 'good') goodIds.add(k); if (v === 'bad') badIds.add(k) })
      } else {
        for (const [k, v] of Object.entries(labels)) { if (v === 'good') goodIds.add(k); if (v === 'bad') badIds.add(k) }
      }
    }
    const posItems = items.filter(i => goodIds.has(i.id))
    const negItems = items.filter(i => badIds.has(i.id))
    function dist(a, b) { const pa = posOriginal(a), pb = posOriginal(b); return Math.hypot((pa.x - pb.x), (pa.y - pb.y)) }
    function minDist(pt, arr) { if (!arr.length) return Infinity; let m = Infinity; for (const s of arr) { const d = dist(pt, s); if (d < m) m = d } return m }
    const coords = {}
    let maxDp = 0, maxDn = 0
    for (const it of items) { const dp = minDist(it, posItems); const dn = minDist(it, negItems); if (isFinite(dp) && dp > maxDp) maxDp = dp; if (isFinite(dn) && dn > maxDn) maxDn = dn }
    function clamp01(v) { return Math.max(0, Math.min(1, v)) }
    for (const it of items) {
      const dp = minDist(it, posItems)
      const dn = minDist(it, negItems)
      let s = 0.5
      if (posItems.length && negItems.length) s = clamp01(dn / (dn + dp + 1e-9))
      else if (posItems.length && !negItems.length) s = clamp01(1 - (dp / (maxDp + 1e-9)))
      else if (negItems.length && !posItems.length) s = clamp01(dn / (maxDn + 1e-9))
      coords[it.id] = s
    }
    const name = prompt('Name this axis', 'Axis') || 'Axis'
    const id = `axis:labels:${Date.now()}`
    dispatch('create', { id, name, coords })
  }

  // Selected axis ids for X and Y
  export let selectedX = null
  export let selectedY = null

  $: axesById = new Map((axes || []).map(a => [a.id, a]))

  function axisName(id) { return (axesById.get(id)?.name) || '—' }
  function getCoord(id, axisId) {
    const ax = axisId ? axesById.get(axisId) : null
    if (!ax || !ax.coords) return undefined
    const v = ax.coords[id]
    return (typeof v === 'number' && isFinite(v)) ? Math.max(0, Math.min(1, v)) : undefined
  }

  // Hover and zoom state
  let griddingActive = false
  let activeCenterId = null
  let cx = 0.5
  let cy = 0.5
  let vf = viewFrac
  let lastTargetsUpdate = 0
  let posRect = { x0: 0, y0: 0, x1: 1, y1: 1, margin: 0 }

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    cx = Math.min(1, Math.max(0, x))
    cy = Math.min(1, Math.max(0, y))
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
  function zoomIn() { vf = clamp(vf * 0.85, 0.08, 0.8) }
  function zoomOut() { vf = clamp(vf / 0.85, 0.08, 0.8) }
  $: if (griddingActive) { posRect = { x0, y0, x1, y1, margin: hoverMargin }; lastTargetsUpdate = 0 }

  // Viewport rect
  $: viewW = vf
  $: viewH = vf
  $: x0 = Math.max(0, Math.min(1 - viewW, cx - viewW / 2))
  $: y0 = Math.max(0, Math.min(1 - viewH, cy - viewH / 2))
  $: x1 = x0 + viewW
  $: y1 = y0 + viewH
  $: hoverMargin = Math.min(0.02, vf * 0.15)

  // Positions selection helpers
  function posOriginal(it) {
    // Derive x from selected X axis (fallback to item.x), and y from selected Y axis (fallback to item.y)
    const ox = getCoord(it.id, selectedX)
    const oy = getCoord(it.id, selectedY)
    const x = (ox !== undefined) ? ox : Number(it.x ?? 0)
    const y = (oy !== undefined) ? oy : Number(it.y ?? 0)
    return { x, y }
  }
  // Normalized items for the lasso overlay – use the actually rendered positions
  // so selection matches the current (possibly customized) axes projection and packing.
  $: lassoItems = Array.isArray(renderItems)
    ? renderItems.map((it) => ({ id: it.id, x: it.x, y: it.y }))
    : []

  // Animation tracks
  let lastPosMap = new Map() // id -> {x,y}
  let prevTargets = new Map() // id -> {x,y}
  let tracks = new Map() // id -> {id,url,from,to}
  function easeInOutCubic(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2 }
  function lerp(a, b, t) { return a + (b - a) * t }
  function posEqual(a, b, eps = 1e-6) {
    if (!a || !b) return false
    return Math.abs(a.x - b.x) <= eps && Math.abs(a.y - b.y) <= eps
  }
  let startTs = performance.now()
  let nowTs = startTs
  let rafId
  function tickAnim() {
    nowTs = performance.now()
    // update last positions in ORIGINAL coord space (not visual)
    if (tracks && tracks.size > 0) {
      for (const tr of tracks.values()) {
        const x = lerp(tr.from.x, tr.to.x, tNorm)
        const y = lerp(tr.from.y, tr.to.y, tNorm)
        lastPosMap.set(tr.id, { x, y })
      }
    }
    if (griddingActive && (nowTs - lastTargetsUpdate > posUpdateMs)) {
      posRect = { x0, y0, x1, y1, margin: hoverMargin }
      lastTargetsUpdate = nowTs
    }
    rafId = requestAnimationFrame(tickAnim)
  }
  onMount(() => { rafId = requestAnimationFrame(tickAnim) })
  onDestroy(() => { cancelAnimationFrame(rafId) })

  // Local packing for inside grid placement
  function computeLocalPacked(itemsArr, rect, sizePx, spacingScale, wPx, hPx) {
    if (!rect) return new Map()
    const rx0 = rect.x0, ry0 = rect.y0, rx1 = rect.x1, ry1 = rect.y1
    const margin = rect.margin || 0
    const stepPx = Math.max(1, sizePx * spacingScale)
    const sx = Math.max(1e-6, stepPx / Math.max(1, wPx))
    const sy = Math.max(1e-6, stepPx / Math.max(1, hPx))
    const inside = itemsArr.filter((it) => {
      const p = posOriginal(it)
      return p.x >= (rx0 - margin) && p.x <= (rx1 + margin) && p.y >= (ry0 - margin) && p.y <= (ry1 + margin)
    })
    if (!inside.length) return new Map()
    const cx = (rx0 + rx1) / 2, cy = (ry0 + ry1) / 2
    let center = inside[0], best = Infinity
    for (const it of inside) {
      const p = posOriginal(it)
      const dx = p.x - cx, dy = p.y - cy
      const d2 = dx*dx + dy*dy
      if (d2 < best) { best = d2; center = it }
    }
    const base = posOriginal(center)
    function cellToXY(ix, iy) { return { x: base.x + ix * sx, y: base.y + iy * sy } }
    const occ = new Set(); const key = (ix,iy)=> ix+','+iy
    occ.add(key(0,0))
    const ordered = [...inside].sort((a,b)=>{
      const pa = posOriginal(a), pb = posOriginal(b)
      if (a.id===center.id) return -1; if (b.id===center.id) return 1
      const da=(pa.x-base.x)**2+(pa.y-base.y)**2; const db=(pb.x-base.x)**2+(pb.y-base.y)**2
      return da-db
    })
    const out = new Map(); out.set(center.id, { x: base.x, y: base.y })
    for (const it of ordered) {
      if (it.id === center.id) continue
      const p = posOriginal(it)
      let ix = Math.round((p.x - base.x) / sx)
      let iy = Math.round((p.y - base.y) / sy)
      let found = null
      const maxR = inside.length + 4
      for (let r=0; r<=maxR && !found; r++) {
        for (let dx=-r; dx<=r; dx++) {
          for (let dy=-r; dy<=r; dy++) {
            if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue
            const tx = ix + dx, ty = iy + dy
            if (occ.has(key(tx,ty))) continue
            found = { ix: tx, iy: ty }
            break
          }
          if (found) break
        }
      }
      if (!found) { found = { ix, iy } }
      occ.add(key(found.ix, found.iy)); out.set(it.id, cellToXY(found.ix, found.iy))
    }
    return out
  }

  // Derived layout
  $: cellPx = 16
  $: imSize = Math.max(minImagePx, Math.floor(cellPx ))
  $: vfRatio = 0.35 / Math.max(0.08, Math.min(1.0, vf))
  $: insideScale = Math.max(1.5, 1.7 + 1 * (vfRatio - 1))
  $: sizeInside = Math.max(minImagePx, Math.floor(imSize * insideScale))
  $: sizeOutside = Math.max(8, Math.floor(imSize * 0.8))
  $: spacingScale = Math.max(1.5, 1.5 + 0.0 * (vfRatio - 1))

  // Precompute inside ids set (using original positions from axes)
  $: insideIds = new Set(items.filter((it) => {
    const p = posOriginal(it)
    return griddingActive && p.x >= (x0 - hoverMargin) && p.x <= (x1 + hoverMargin) && p.y >= (y0 - hoverMargin) && p.y <= (y1 + hoverMargin)
  }).map((it) => it.id))

  $: localPacked = computeLocalPacked(items, posRect, sizeInside, spacingScale, width, height)

  // Compute animation targets and renderItems
  $: {
    const nextTargets = new Map(); const m = new Map(); let anyChange = false
    for (const it of items) {
      const p = posOriginal(it)
      const rx0 = posRect?.x0 ?? x0, ry0 = posRect?.y0 ?? y0, rx1 = posRect?.x1 ?? x1, ry1 = posRect?.y1 ?? y1
      const rmg = posRect?.margin ?? hoverMargin
      const inside = griddingActive && p.x >= (rx0 - rmg) && p.x <= (rx1 + rmg) && p.y >= (ry0 - rmg) && p.y <= (ry1 + rmg)
      const lp = localPacked.get(it.id)
      const to = (inside && lp) ? lp : p
      const prevTarget = prevTargets.get(it.id)
      if (!posEqual(prevTarget, to)) anyChange = true
      const from = lastPosMap.get(it.id) || p
      m.set(it.id, { id: it.id, url: it.url, from, to })
      nextTargets.set(it.id, to)
    }
    tracks = m
    if (anyChange) startTs = performance.now()
    prevTargets = nextTargets
  }

  $: elapsed = Math.max(0, nowTs - startTs)
  $: raw = Math.min(1, duration > 0 ? elapsed / duration : 1)
  $: tNorm = easeInOutCubic(raw)
  $: renderItems = Array.from(tracks.values()).map((tr) => ({ id: tr.id, url: tr.url, x: toVis(lerp(tr.from.x, tr.to.x, tNorm)), y: toVis(lerp(tr.from.y, tr.to.y, tNorm)) }))

  function clamp01(v) { return Math.max(0, Math.min(1, v)) }
  $: gridBackdrop = (function computeBackdrop(active, list, insideSet, sizePx, wPx, hPx) {
    if (!active || !Array.isArray(list) || list.length === 0) return null
    const halfWn = (sizePx / Math.max(1, wPx)) / 2
    const halfHn = (sizePx / Math.max(1, hPx)) / 2
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity
    let count = 0
    for (const it of list) {
      if (!insideSet.has(it.id)) continue
      count++
      if (it.x < minx) minx = it.x
      if (it.y < miny) miny = it.y
      if (it.x > maxx) maxx = it.x
      if (it.y > maxy) maxy = it.y
    }
    if (count === 0 || !isFinite(minx)) return null
    // list is already in visual space; extend by half image size (normalized)
    const x0b = clamp01(minx - halfWn)
    const y0b = clamp01(miny - halfHn)
    const x1b = clamp01(maxx + halfWn)
    const y1b = clamp01(maxy + halfHn)
    return { x0: x0b, y0: y0b, w: Math.max(0, x1b - x0b), h: Math.max(0, y1b - y0b) }
  })(griddingActive, renderItems, insideIds, sizeInside, width, height)

  function pickOrToggle(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = (e.clientX - rect.left) / rect.width
    const py = (e.clientY - rect.top) / rect.height
    let hitId = null
    for (let i = renderItems.length - 1; i >= 0; i--) {
      const it = renderItems[i]
      const inside = griddingActive && insideIds.has(it.id)
      const sz = inside ? sizeInside : sizeOutside
      const dx = Math.abs(px - it.x) * rect.width
      const dy = Math.abs(py - it.y) * rect.height
      if (dx <= sz / 2 && dy <= sz / 2) { hitId = it.id; break }
    }
    if (hitId) {
      griddingActive = true
      activeCenterId = hitId
      const orig = items.find(j => j.id === hitId)
      if (orig) {
        const p = posOriginal(orig); cx = p.x; cy = p.y
      }
      posRect = { x0, y0, x1, y1, margin: hoverMargin }
      lastTargetsUpdate = 0
    } else {
      griddingActive = false
      activeCenterId = null
      posRect = { x0: 0, y0: 0, x1: 1, y1: 1, margin: 0 }
      lastTargetsUpdate = 0
    }
  }

  // Drag and drop handlers for X/Y axis selectors
  function allowDrop(e) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }
  function onDropX(e) {
    e.preventDefault()
    const id = e.dataTransfer.getData('application/axis-id') || e.dataTransfer.getData('text/plain')
    if (!id) return
    selectedX = id
    dispatch('axesChange', { selectedX, selectedY })
  }
  function onDropY(e) {
    e.preventDefault()
    const id = e.dataTransfer.getData('application/axis-id') || e.dataTransfer.getData('text/plain')
    if (!id) return
    selectedY = id
    dispatch('axesChange', { selectedX, selectedY })
  }
  function clearX() { selectedX = null; dispatch('axesChange', { selectedX, selectedY }) }
  function clearY() { selectedY = null; dispatch('axesChange', { selectedX, selectedY }) }
</script>

<!-- Y axis selector will be positioned inside the minimap (left side) -->

<div class="tile"><div class="tile-content flush">
  <div
    role="img"
    aria-label="Axes minimap"
    class="minimap"
    style={`width:${width}px;height:${height}px;`}
    on:mousemove={onMove}
    on:click|stopPropagation={(e) => pickOrToggle(e)}
  >
  {#if griddingActive && gridBackdrop}
    <div
      class="absolute pointer-events-none rounded"
      style={`left:${gridBackdrop.x0 * 100}%;top:${gridBackdrop.y0 * 100}%;width:${gridBackdrop.w * 100}%;height:${gridBackdrop.h * 100}%;background:rgba(229,231,235,0.5);z-index:5;`}
    />
  {/if}
  {#each renderItems as it (it.id)}
    <img
      alt=""
      src={it.url}
      class="absolute object-cover rounded"
      style={`left:${it.x * 100}%;top:${it.y * 100}%;transform:translate(-50%,-50%);width:${(griddingActive && insideIds.has(it.id) ? sizeInside : sizeOutside)}px;height:${(griddingActive && insideIds.has(it.id) ? sizeInside : sizeOutside)}px;transition:width 120ms ease,height 120ms ease; z-index:${(griddingActive && insideIds.has(it.id) ? 10 : 1)}; opacity:${(griddingActive && insideIds.has(it.id) ? 1 : 0.8)}; border:${labelOf(it.id)?'2px solid '+(labelOf(it.id)==='good'?'#16a34a':'#dc2626'):'none'}; box-shadow:${labelOf(it.id)?'0 0 0 1px rgba(255,255,255,0.8)':'none'};`}
      loading="lazy"
    />
  {/each}

  {#if griddingActive}
    <div
      class="absolute border border-blue-500/70 pointer-events-none"
      style={`left:${toVis(x0) * 100}%;top:${toVis(y0) * 100}%;width:${(viewW * (1 - 2*displayMargin)) * 100}%;height:${(viewH * (1 - 2*displayMargin)) * 100}%;`}
    />
  {/if}

  <div class="toolbar pos-top-right right-just z-10">
    <button type="button" class="btn btn-icon btn-minimap" on:click|stopPropagation={zoomIn} aria-label="Zoom in">+</button>
    <button type="button" class="btn btn-icon btn-minimap" on:click|stopPropagation={zoomOut} aria-label="Zoom out">−</button>
  </div>

  <div class="toolbar pos-top-left z-10">
    <button type="button" class={`btn btn-icon btn-minimap ${lassoEnabled?'btn-primary':''}`} on:click|stopPropagation={() => { lassoEnabled = !lassoEnabled; if (lassoEnabled) { try { lassoRef && lassoRef.reset && lassoRef.reset() } catch(_) {} } }} aria-label="Toggle lasso">
      <!-- simple lasso icon -->
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 10c0-3.314 3.582-6 8-6s8 2.686 8 6-3.582 6-8 6c-1.4 0-2.7-.26-3.8-.72L6 16l.4-2.2C5.2 12.7 4 11.5 4 10z"/></svg>
    </button>
  </div>

  <!-- Y axis control on the left, fully vertical (label + rotated select) -->
  <div class="axis-rail-left text-sm"
       on:dragover={allowDrop} on:drop={onDropY} title="Drop an axis here">
    <div class="text-gray-700" style="">Y axis</div>
    <div class="origin-top-left" style="">
      <select class="text-sm" on:change={(e)=>{ selectedY = e.currentTarget.value || null; dispatch('axesChange', { selectedX, selectedY }) }}>
        <option value="">(none)</option>
        {#each axes as ax}
          <option value={ax.id} selected={selectedY===ax.id}>{ax.name}</option>
        {/each}
      </select>
    </div>
  </div>

  <!-- Bottom-right create selection button -->
  <div class="toolbar pos-bottom-right right-just">
    <button class="btn btn-sm btn-minimap" on:click|stopPropagation={() => {
      const good = []
      const bad = []
      try {
        if (labels instanceof Map) { labels.forEach((v,k)=>{ if (v==='good') good.push(k); else if (v==='bad') bad.push(k) }) }
        else { for (const [k,v] of Object.entries(labels||{})) { if (v==='good') good.push(k); else if (v==='bad') bad.push(k) } }
      } catch(_) {}
      const name = prompt('Name this selection', 'Selection') || 'Selection'
      dispatch('saveSelection', { id: `sel:${Date.now()}`, name, posIds: good, negIds: bad, active: true })
    }}>Create selection</button>
  </div>

  <!-- Bottom-left info callout -->
  <div class="absolute left-1 bottom-1 z-10" style="max-width:260px">
    <Callout storageKey="minimap" variant="info" title="Minimap">
      Zoom with +/-, toggle lasso to label regions, then save selections.
    </Callout>
  </div>

  {#if lassoEnabled}
    <div class="absolute top-1 left-10 z-10 bg-white/90 rounded shadow px-2 py-1 text-sm flex items-center gap-2">
      <button class={`btn btn-xs ${lassoMode==='pos'?'btn-success':''}`} on:click|stopPropagation={() => setLassoMode('pos')} aria-label={`Positive (${posCount})`}>
        <span class="i-heroicons-hand-thumb-up" /> Positive ({posCount})
      </button>
      <button class={`btn btn-xs ${lassoMode==='neg'?'btn-danger':''}`} on:click|stopPropagation={() => setLassoMode('neg')} aria-label={`Negative (${negCount})`}>
        <span class="i-heroicons-hand-thumb-down" /> Negative ({negCount})
      </button>
      <button class="btn btn-xs btn-primary" on:click|stopPropagation={createAxisFromLabels} aria-label="Create axis">
        <span class="i-heroicons-plus-circle" /> Create axis
      </button>
      <button class="btn btn-xs btn-ui-secondary" on:click|stopPropagation={() => {
        const updates = []
        try {
          if (labels instanceof Map) { labels.forEach((_,k)=> updates.push({ id: k, label: null })) }
          else { for (const k of Object.keys(labels||{})) updates.push({ id: k, label: null }) }
        } catch(_) {}
        if (updates.length) dispatch('label', { updates })
        try { lassoRef && lassoRef.reset && lassoRef.reset() } catch(_) {}
      }} aria-label="Clear all labels">Clear</button>
    </div>
    
    <LassoSelector
      bind:this={lassoRef}
      enabled={true}
      width={width}
      height={height}
      items={lassoItems}
      strokeColor={lassoMode==='pos' ? '#16a34a' : '#dc2626'}
      fillColor={lassoMode==='pos' ? 'rgba(22,163,74,0.12)' : 'rgba(220,38,38,0.12)'}
      on:select={onLassoSelect}
    />
  {/if}
    <!-- X axis selector toolbar (bottom-center of minimap) -->
    <div class="axis-rail-bottom text-sm">
      <div class="inline-flex items-center gap-2" on:dragover={allowDrop} on:drop={onDropX} title="Drop an axis here">
        <span class="text-gray-700 text-sm">X axis</span>
        <select class="text-sm" on:change={(e)=>{ selectedX = e.currentTarget.value || null; dispatch('axesChange', { selectedX, selectedY }) }}>
          <option value="">(none)</option>
          {#each axes as ax}
            <option value={ax.id} selected={selectedX===ax.id}>{ax.name}</option>
          {/each}
        </select>
      </div>
    </div>
  </div>
</div></div>

<style>
</style>
