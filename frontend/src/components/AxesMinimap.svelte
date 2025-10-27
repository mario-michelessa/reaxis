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
  export let duration = 300 // ms
  export let minImagePx = 25
  export let posUpdateMs = 450

  const dispatch = createEventDispatcher()

  // Visual margin for display (map [0,1] -> [m, 1-m])
  export let displayMargin = 0.1
  function toVis(v) {
    const m = Math.max(0, Math.min(0.49, Number(displayMargin || 0)))
    const nv = Number(v || 0)
    return m + nv * (1 - 2 * m)
  }
  function fromVis(v) {
    const m = Math.max(0, Math.min(0.49, Number(displayMargin || 0)))
    const nv = Number(v || 0)
    const denom = Math.max(1e-6, (1 - 2 * m))
    return (nv - m) / denom
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
      if (labels instanceof Map) { labels.forEach((v) => { if (v === 'pos') c++ }) }
      else { for (const v of Object.values(labels)) if (v === 'pos') c++ }
    } catch (_) {}
    return c
  })()
  $: negCount = (() => {
    let c = 0
    if (!labels) return 0
    try {
      if (labels instanceof Map) { labels.forEach((v) => { if (v === 'neg') c++ }) }
      else { for (const v of Object.values(labels)) if (v === 'neg') c++ }
    } catch (_) {}
    return c
  })()

  // Lasso selection & labeling
  let lassoEnabled = true
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
        if (cur === 'pos') updates.push({ id, label: null }) // toggle off
        else updates.push({ id, label: 'pos' }) // set or flip to pos
      } else {
        if (cur === 'neg') updates.push({ id, label: null })
        else updates.push({ id, label: 'neg' })
      }
    }
    dispatch('label', { updates })
    // Clear selection and lasso path; keep lasso enabled for next selection
    lastSelectedIds = []
    try { lassoRef && lassoRef.reset && lassoRef.reset() } catch(_) {}
    // Suppress grid toggling immediately after a lasso selection completes
    suppressUntil = performance.now() + 400
  }
  // Buttons now only change mode (color); application happens on release
  function setLassoMode(kind) { lassoMode = (kind === 'neg') ? 'neg' : 'pos' }
  function createAxisFromLabels() {
    // Build axis scores based on current labels
    const posIds = new Set()
    const negIds = new Set()
    if (labels) {
      if (labels instanceof Map) {
        labels.forEach((v, k) => { if (v === 'pos') posIds.add(k); if (v === 'neg') negIds.add(k) })
      } else {
        for (const [k, v] of Object.entries(labels)) { if (v === 'pos') posIds.add(k); if (v === 'neg') negIds.add(k) }
      }
    }
    const posItems = items.filter(i => posIds.has(i.id))
    const negItems = items.filter(i => negIds.has(i.id))
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
  let gridRect = null // fixed rect while grid is active
  let griddedImsRect = null // last rect used to compute gridded images
  let gridMargin = 0.01
  // Zoom center for scaling points
  let cx = 0.5
  let cy = 0.5
  // Rectangle center follows cursor
  let rcx = 0.5
  let rcy = 0.5
  let vf = viewFrac
  let lastTargetsUpdate = 0
  let posRect = { x0: 0, y0: 0, x1: 1, y1: 1, margin: 0 }

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const vx = (e.clientX - rect.left) / rect.width
    const vy = (e.clientY - rect.top) / rect.height
    rcx = fromVis(vx)
    rcy = fromVis(vy)
  }
  let zoomZ = 1.0 // visual zoom scale around (cx,cy); 1 = no zoom
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
  // Buttons: resize the viewport (blue rect)
  function viewportZoomIn() { vf = clamp(vf * 0.88, 0.04, 0.9) }
  function viewportZoomOut() { vf = clamp(vf / 0.88, 0.04, 0.9) }
  // Wheel: zoom content around center
  function contentZoomIn() { zoomZ = clamp(zoomZ * 1.12, 1.0, 6.0) }
  function contentZoomOut() { zoomZ = clamp(zoomZ / 1.12, 1.0, 6.0) }
  function resetView() { cx = 0.5; cy = 0.5; zoomZ = 1.0; griddingActive = false; activeCenterId = null; gridRect = null; griddedImsRect = null; posRect = { x0: 0, y0: 0, x1: 1, y1: 1, margin: 0 } }
  function onWheel(e) {
    // Zoom around the cursor so the pointed spot stays fixed
    const rect = e.currentTarget.getBoundingClientRect()
    const vx = (e.clientX - rect.left) / Math.max(1, rect.width)
    const vy = (e.clientY - rect.top) / Math.max(1, rect.height)
    const zx = fromVis(vx)
    const zy = fromVis(vy)
    const factor = 1.12
    const nextZ = clamp(e.deltaY < 0 ? (zoomZ * factor) : (zoomZ / factor), 1.0, 6.0)
    if (nextZ === zoomZ) return
    // Convert current cursor zoom-space position (zx,zy) to world (pre-zoom) coords
    const px = cx + (zx - cx) / Math.max(1e-6, zoomZ)
    const py = cy + ((1 - zy) - cy) / Math.max(1e-6, zoomZ)
    if (Math.abs(1 - nextZ) > 1e-6) {
      const denom = (1 - nextZ)
      cx = clamp((zx - nextZ * px) / denom, 0, 1)
      cy = clamp(((1 - zy) - nextZ * py) / denom, 0, 1)
    }
    zoomZ = nextZ
  }

  // Viewport rect
  $: viewW = vf
  $: viewH = vf
  $: x0 = Math.max(0, Math.min(1 - viewW, rcx - viewW / 2))
  $: y0 = Math.max(0, Math.min(1 - viewH, rcy - viewH / 2))
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
  $: lassoItems = Array.isArray(renderItemsVisible)
    ? renderItemsVisible.map((it) => ({ id: it.id, x: it.x, y: 1 - it.y }))
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
      return p.x >= (rx0 - margin) && p.x <= (rx1 + margin) && (1 - p.y) >= (ry0 - margin) && (1 - p.y) <= (ry1 + margin)
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
  $: spacingScale = Math.max(1.3, 1.3 + 0.0 * (vfRatio - 1))

  // Precompute inside ids set (using original positions and fixed gridRect captured on click)
  $: insideIds = new Set(itemsFiltered.filter((it) => {
    if (!griddingActive || !gridRect) return false
    const p = posOriginal(it)
    return p.x >= (gridRect.x0 - hoverMargin) && p.x <= (gridRect.x1 + hoverMargin) && (1 - p.y) >= (gridRect.y0 - hoverMargin) && (1 - p.y) <= (gridRect.y1 + hoverMargin)
  }).map((it) => it.id))

  $: localPacked = computeLocalPacked(itemsFiltered, (griddingActive && gridRect) ? gridRect : null, sizeInside, spacingScale, width, height)

  // Compute animation targets and renderItems
  $: {
    const nextTargets = new Map(); const m = new Map(); let anyChange = false
    for (const it of itemsFiltered) {
      const p = posOriginal(it)
      const rect = (griddingActive && gridRect) ? gridRect : null
      const rx0 = rect?.x0 ?? x0, ry0 = rect?.y0 ?? y0, rx1 = rect?.x1 ?? x1, ry1 = rect?.y1 ?? y1
      const rmg = rect?.margin ?? hoverMargin
      const inside = !!(griddingActive && rect && p.x >= (rx0 - rmg) && p.x <= (rx1 + rmg) && (1 - p.y) >= (ry0 - rmg) && (1 - p.y) <= (ry1 + rmg))
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
  $: renderItems = Array.from(tracks.values()).map((tr) => {
    const wx = lerp(tr.from.x, tr.to.x, tNorm)
    const wy = lerp(tr.from.y, tr.to.y, tNorm)
    const zx = cx + (wx - cx) * zoomZ
    const zy = cy + (wy - cy) * zoomZ
    const sx = toVis(zx)
    const sy = toVis(zy)
    return { id: tr.id, url: tr.url, x: sx, y: sy }
  })
  $: renderItemsVisible = Array.isArray(renderItems) ? renderItems.filter((it) => it.x >= 0 && it.x <= 1 && it.y >= 0 && it.y <= 1) : []
  $: zoomScale = zoomZ

  function clamp01(v) { return Math.max(0, Math.min(1, v)) }
  // Grey rectangle based on visible gridded images (static while grid is active)
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
      if (1-it.y < miny) miny = 1-it.y
      if (it.x > maxx) maxx = it.x
      if (1-it.y > maxy) maxy = 1-it.y
    }
    if (count === 0 || !isFinite(minx)) return null
    const x0b = clamp01(minx - halfWn)
    const y0b = clamp01(miny - halfHn)
    const x1b = clamp01(maxx + halfWn)
    const y1b = clamp01(maxy + halfHn)
    return { x0: x0b, y0: y0b, w: Math.max(0, x1b - x0b), h: Math.max(0, y1b - y0b), x1: x1b, y1: y1b }
  })(griddingActive, renderItemsVisible, insideIds, Math.floor(sizeInside * zoomScale), width, height)

  let suppressUntil = 0
  function pickOrToggle(e) {
    if (zoomItemId) { return } // When overlay is open, ignore background clicks
    if (performance.now() < suppressUntil) { return }
    const rect = e.currentTarget.getBoundingClientRect()
    const px = (e.clientX - rect.left) / rect.width
    const py = (e.clientY - rect.top) / rect.height
    
    // Find nearest visible image under cursor (in visual coords)
    let hitId = null
    for (let i = renderItemsVisible.length - 1; i >= 0; i--) {
      const it = renderItemsVisible[i]
      const inside = griddingActive && insideIds.has(it.id)
      const sz = inside ? sizeInside : sizeOutside
      const dx = Math.abs(px - it.x) * rect.width
      const dy = Math.abs(py - 1 + it.y) * rect.height
      if (dx <= sz / 2 && dy <= sz / 2) { hitId = it.id; break }
    }
    // If grid is active, handle inside/outside clicks
    if (griddingActive && gridRect) {
      const vx = px
      const vy = py
      // Compare against gridded images rect for interaction; fallback to captured rect if missing
      const rx0 = gridBackdrop ? gridBackdrop.x0 : toVis(gridRect.x0)
      const ry0 = gridBackdrop ? gridBackdrop.y0 : toVis(gridRect.y0)
      const rx1 = gridBackdrop ? gridBackdrop.x1 : toVis(gridRect.x1)
      const ry1 = gridBackdrop ? gridBackdrop.y1 : toVis(gridRect.y1)
      const inside = (vx >= rx0 && vx <= rx1 && vy >= ry0 && vy <= ry1)
      if (!inside) {
        // Click outside grid -> close grid
        griddingActive = false
        activeCenterId = null
        gridRect = null
        posRect = { x0: 0, y0: 0, x1: 1, y1: 1, margin: 0 }
        lastTargetsUpdate = 0
        try { console.log('[minimap] grid closed') } catch(_) {}
        return
      }
      // Inside grid: if clicking a gridded image, open zoom overlay
      if (hitId && insideIds && insideIds.has(hitId)) {
        zoomItemId = hitId
        try { console.log('[minimap] zoom image', hitId) } catch(_) {}
        return
      }
      // Otherwise do nothing (keep grid)
      return
    }
    // Grid not active: enable grid on current viewport rectangle (capture current blue rect)
    griddingActive = true
    activeCenterId = null
    gridRect = { x0, y0, x1, y1, margin: hoverMargin }
    lastTargetsUpdate = 0
    try {
      console.log('[minimap] grid enable gridRect (norm)', gridRect)
      // compute and log grey bounds based on current visible items
      const sz = Math.floor(sizeInside * zoomScale)
      const halfWn = (sz / Math.max(1, width)) / 2
      const halfHn = (sz / Math.max(1, height)) / 2
      let minx=Infinity, miny=Infinity, maxx=-Infinity, maxy=-Infinity, count=0
      for (const it of renderItemsVisible) { if (insideIds.has(it.id)) { count++; if (it.x<minx) minx=it.x; if (it.y<miny) miny=it.y; if (it.x>maxx) maxx=it.x; if (it.y>maxy) maxy=it.y } }
      if (count>0 && isFinite(minx)) {
        const x0b = clamp01(minx - halfWn), y0b = clamp01(miny - halfHn)
        const x1b = clamp01(maxx + halfWn), y1b = clamp01(maxy + halfHn)
        console.log('[minimap] grey visual %', { x0: +(x0b*100).toFixed(1), y0: +(y0b*100).toFixed(1), x1: +(x1b*100).toFixed(1), y1: +(y1b*100).toFixed(1) })
      }
      console.log('[minimap] insideIds count', insideIds.size, 'sample', Array.from(insideIds).slice(0,10))
    } catch(_) {}
  }

  
  let zoomItemId = null
  function labelZoomed(kind) {
    const id = zoomItemId
    if (!id) return
    const updates = [{ id, label: kind === 'pos' ? 'pos' : 'neg' }]
    dispatch('label', { updates })
    zoomItemId = null
  }
  function closeZoom() { zoomItemId = null; suppressUntil = performance.now() + 250 }

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

  // Selections provided by parent and top-center filtering controls
  export let selections = [] // [{id,name,posIds,negIds,active}]
  $: selectionsById = new Map((selections||[]).map(s => [s.id, s]))
  // Top-center filter dropdown + drag-and-drop of selections
  // Modes: 'all' shows everything; 'keep-pos' shows only selected selection's positives;
  //        'discard-neg' hides selected selection's negatives (keeps non-negative images)
  let filterMode = 'all' // 'all' | 'keep-pos' | 'discard-neg'
  let filterSelection = null // { id, name, posIds:[], negIds:[] }
  let selectedSelectionId = ''
  $: filterPosSet = new Set(Array.isArray(filterSelection?.posIds) ? filterSelection.posIds : [])
  $: filterNegSet = new Set(Array.isArray(filterSelection?.negIds) ? filterSelection.negIds : [])
  // Keep filter in sync with current selections list (e.g., if deleted)
  $: if (filterSelection && selectionsById.size > 0) {
    if (selectionsById.has(filterSelection.id)) {
      // Update reference to the canonical selection object
      filterSelection = selectionsById.get(filterSelection.id)
    } else if (selectedSelectionId) {
      // Selected id is no longer present
      selectedSelectionId = ''
      filterSelection = null
      filterMode = 'all'
    }
  }
  function onFilterModeChange(val) {
    // If no selection is set, non-'all' modes are not applicable
    if ((val === 'keep-pos' || val === 'discard-neg') && !filterSelection) {
      filterMode = 'all'
      return
    }
    // Switching manually clears selection details when returning to 'all'
    filterMode = val
    if (filterMode === 'all') filterSelection = null
  }
  function allowDropSelection(e) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }
  function onDropSelection(e) {
    e.preventDefault()
    try {
      const raw = e.dataTransfer.getData('application/x-selection') || e.dataTransfer.getData('text/plain')
      if (!raw) return
      const sel = JSON.parse(raw)
      // Prefer the canonical selection from props when available
      const viaId = sel?.id && selectionsById.get(sel.id)
      const s = viaId || sel
      selectedSelectionId = viaId ? viaId.id : ''
      applySelectionChoice(s)
    } catch (_) { /* ignore */ }
  }
  function onDropdownSelect(id) {
    selectedSelectionId = id || ''
    if (!id) {
      filterSelection = null
      filterMode = 'all'
      return
    }
    const sel = selectionsById.get(id)
    applySelectionChoice(sel)
  }
  function applySelectionChoice(sel) {
    if (!sel) { filterSelection = null; filterMode = 'all'; return }
    const pos = Array.isArray(sel?.posIds) ? sel.posIds : []
    const neg = Array.isArray(sel?.negIds) ? sel.negIds : []
    filterSelection = sel
    if (pos.length > 0 && neg.length === 0) {
      filterMode = 'keep-pos'
    } else if (neg.length > 0 && pos.length === 0) {
      filterMode = 'discard-neg'
    } else if (pos.length > 0 && neg.length > 0) {
      // Mixed: wait for explicit button click; show controls
      filterMode = 'all'
    } else {
      // Empty selection: show all
      filterMode = 'all'
    }
  }

  // Items used for rendering after applying filter
  $: itemsFiltered = (function() {
    const base = Array.isArray(items) ? items : []
    if (filterMode === 'keep-pos') return base.filter(it => filterPosSet.has(it.id))
    if (filterMode === 'discard-neg') return base.filter(it => !filterNegSet.has(it.id))
    return base
  })()
</script>

<!-- Y axis selector will be positioned inside the minimap (left side) -->

<div class="tile"><div class="tile-content flush">
  <div
    role="img"
    aria-label="Axes minimap"
    class="minimap"
    style={`width:${width}px;height:${height}px;`}
    on:mousemove={onMove}
    on:wheel|stopPropagation|preventDefault={onWheel}
    on:click|stopPropagation|preventDefault={(e) => pickOrToggle(e)}
  >
  {#if griddingActive && gridBackdrop}
    <div
      class="absolute pointer-events-none rounded"
      style={`left:${(gridBackdrop.x0 - gridMargin) * 100}%;top:${(gridBackdrop.y0 - gridMargin) * 100}%;width:${(gridBackdrop.w + 2*gridMargin) * 100}%;height:${(gridBackdrop.h + 2*gridMargin) * 100}%;background:rgba(229,231,235,0.9);z-index:5; transition: opacity 200ms ease; opacity:1;`}
    />
    {/if}
  {#each renderItemsVisible as it (it.id)}
    <img
      alt=""
      src={it.url}
      class="absolute object-cover rounded"
      decoding="async"
      fetchpriority="low"
      style={`left:${it.x * 100}%;
              top:${(1 - it.y) * 100}%;
              transform:translate(-50%,-50%);
              width:${Math.floor((griddingActive && insideIds.has(it.id) ? sizeInside : sizeOutside) * zoomScale)}px;
              height:${Math.floor((griddingActive && insideIds.has(it.id) ? sizeInside : sizeOutside) * zoomScale)}px;
              transition:width 120ms ease,height 120ms ease; 
              z-index:${(griddingActive && insideIds.has(it.id) ? 10 : 1)}; 
              opacity:${(griddingActive && insideIds.has(it.id) ? 1 : 0.85)}; 
              border:${labelOf(it.id)?'2px solid '+(labelOf(it.id)==='pos'?'#16a34a':'#dc2626'):'none'}; 
              box-shadow:${labelOf(it.id)?'0 0 0 1px rgba(255,255,255,0.8)':'none'};`}
    />
  {/each}

  {#if true}
    <div
      class="absolute border border-blue-500/70 pointer-events-none"
      style={`left:${toVis(x0) * 100}%;top:${toVis(y0) * 100}%;width:${(toVis(x1)-toVis(x0)) * 100}%;height:${(toVis(y1)-toVis(y0)) * 100}%;`}
    />
  {/if}

  

  {#if zoomItemId}
    <div class="absolute inset-0 bg-black/40 flex items-center justify-center z-30" on:click|stopPropagation={closeZoom}>
      <div class="bg-white rounded shadow-lg p-3 relative" on:click|stopPropagation style="max-width:90%;max-height:85%;">
        {#each renderItemsVisible.filter(r => r.id===zoomItemId) as itz}
          <img alt="zoom" src={itz.url} style="max-width:80vw; max-height:70vh; object-fit:contain; display:block; margin:auto;" />
        {/each}
        <div class="mt-2 flex gap-2 justify-center z-40 relative">
          <button class="btn btn-sm btn-success" style="z-index:41" on:click|stopPropagation|preventDefault={() => labelZoomed('pos')}><span class="i-heroicons-hand-thumb-up" /> Label as positive</button>
          <button class="btn btn-sm btn-danger" style="z-index:41" on:click|stopPropagation|preventDefault={() => labelZoomed('neg')}><span class="i-heroicons-hand-thumb-down" /> Label as negative</button>
          <button class="btn btn-sm btn-ui-secondary" style="z-index:41" on:click|stopPropagation|preventDefault={closeZoom}>Close</button>
        </div>
      </div>
    </div>
  {/if}

  <div class="toolbar pos-top-right right-just z-10">
    <button type="button" class="btn btn-icon btn-minimap" on:click|stopPropagation|preventDefault={viewportZoomIn} aria-label="Zoom in">+</button>
    <button type="button" class="btn btn-icon btn-minimap" on:click|stopPropagation|preventDefault={viewportZoomOut} aria-label="Zoom out">−</button>
    <button type="button" class="btn btn-icon btn-minimap" on:click|stopPropagation|preventDefault={resetView} title="Reset view" aria-label="Reset view">
      ⟲
    </button>
  </div>

  <div class="toolbar pos-top-left z-10">
    <button type="button" class={`btn btn-icon btn-minimap ${lassoEnabled?'btn-primary':''}`} on:click|stopPropagation={() => { lassoEnabled = !lassoEnabled; if (lassoEnabled) { try { lassoRef && lassoRef.reset && lassoRef.reset() } catch(_) {} } }} aria-label="Toggle lasso">
      <!-- simple lasso icon -->
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 10c0-3.314 3.582-6 8-6s8 2.686 8 6-3.582 6-8 6c-1.4 0-2.7-.26-3.8-.72L6 16l.4-2.2C5.2 12.7 4 11.5 4 10z"/></svg>
    </button>
  </div>

  <!-- Top-center filter dropdown and drop target -->
  <div class="axis-rail-top text-sm" on:dragover={allowDropSelection} on:drop={onDropSelection} title="Drop a selection here or choose one to filter">
    <div class="inline-flex items-center gap-2">
      <span class="text-gray-700 text-sm">Filter</span>
      <select class="text-sm" on:change={(e)=> onDropdownSelect(e.currentTarget.value)}>
        <option value="" selected={!selectedSelectionId}>All images</option>
        {#each (selections||[]) as s}
          <option value={s.id} selected={selectedSelectionId===s.id}>{s.name || s.id}</option>
        {/each}
      </select>
      {#if filterSelection && (Array.isArray(filterSelection.posIds) && filterSelection.posIds.length>0) && (Array.isArray(filterSelection.negIds) && filterSelection.negIds.length>0)}
        <button class="btn btn-xs btn-positive" on:click={() => { filterMode='keep-pos' }} title="Keep only positive examples">Keep positive</button>
        <button class="btn btn-xs btn-negative" on:click={() => { filterMode='discard-neg' }} title="Remove negative examples">Remove negatives</button>
      {/if}
      {#if filterMode!=='all' && filterSelection}
        <button class="btn btn-xs btn-ui-secondary" on:click={() => { filterMode='all'; filterSelection=null; selectedSelectionId='' }}>Clear</button>
      {/if}
    </div>
  </div>

  <!-- Y axis control on the left, fully vertical (label + rotated select) -->
  <div class="axis-rail-left text-sm left-just" on:dragover={allowDrop} on:drop={onDropY} title="Drop an axis here">
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
  <div class="toolbar pos-bottom-right right-just z-10">
    <button class="btn btn-sm btn-minimap" on:click|stopPropagation={() => {
      const pos = []
      const neg = []
      try {
        if (labels instanceof Map) { labels.forEach((v,k)=>{ if (v==='pos') pos.push(k); else if (v==='neg') neg.push(k) }) }
        else { for (const [k,v] of Object.entries(labels||{})) { if (v==='pos') pos.push(k); else if (v==='neg') neg.push(k) } }
      } catch(_) {}
      const name = prompt('Name this selection', 'Selection') || 'Selection'
      dispatch('saveSelection', { id: `sel:${Date.now()}`, name, posIds: pos, negIds: neg, active: true })
    }}>Create selection</button>
  </div>

  <!-- Bottom-left info callout -->
  <div class="absolute left-1 bottom-1 z-10" style="max-width:260px">
    <Callout storageKey="minimap" variant="info" title="Define concepts">
      Click to zoom on images. Lasso by dragging to label images as positive or negative. Use labeled images to create new axes or selections.
    </Callout>
  </div>

  {#if lassoEnabled}
    <div class="absolute top-10 left-1 z-10 bg-white/90 rounded shadow px-2 py-1 text-sm flex flex-col items-stretch gap-1">
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
