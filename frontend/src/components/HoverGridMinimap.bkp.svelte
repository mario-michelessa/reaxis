<script>
  // HoverGridMinimap: shows all items as points at original coords.
  // When hovering, items inside the viewport square animate to their grid coords
  // and render as images directly inside the minimap. Upon leaving, they animate back to original.

  export let items = [] // [{ id, url, x, y, gx, gy }]
  export let width = 700
  export let height = 700
  export let viewFrac = 0.35 // viewport square fraction (initial)
  export let gridSize = 0 // used to size images
  export let duration = 400 // ms
  export let easing = 'easeInOutCubic'
  export let minImagePx = 12
  // Experimental: compute local packing inside hover rect instead of backend grid
  export let experimentalLocalPacking = false

  let hovering = false
  let cx = 0.5
  let cy = 0.5
  // Internal zoom state controls square size
  let vf = viewFrac
  // Throttle for retargeting to reduce flicker
  export let posUpdateMs = 450
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

  // Position helpers
  function posFrom(it, which) {
    if (!it) return { x: 0, y: 0 }
    if (which === 'grid') return { x: Number(it.gx ?? it.x ?? 0), y: Number(it.gy ?? it.y ?? 0) }
    return { x: Number(it.x ?? 0), y: Number(it.y ?? 0) }
  }

  // Viewport rect based on center and fraction
  $: viewW = vf
  $: viewH = vf
  $: x0 = Math.max(0, Math.min(1 - viewW, cx - viewW / 2))
  $: y0 = Math.max(0, Math.min(1 - viewH, cy - viewH / 2))
  $: x1 = x0 + viewW
  $: y1 = y0 + viewH
  // Soft margin around the hover window to reduce boundary flicker
  $: hoverMargin = Math.min(0.02, vf * 0.15)

  // Note: do not hide dependencies behind a function; reference x0/y0/hovering
  // directly in reactive blocks so Svelte re-runs when hover window changes.
  function inWindowOrig(it) {
    if (!hovering) return false
    const x = Number(it.x ?? 0)
    const y = Number(it.y ?? 0)
    return x >= (x0 - hoverMargin) && x <= (x1 + hoverMargin) && y >= (y0 - hoverMargin) && y <= (y1 + hoverMargin)
  }

  // Animation tracks
  // - lastPosMap: last rendered position for each item (mutated in RAF, no reassignment)
  // - prevTargets: last target position
  let lastPosMap = new Map() // id -> {x,y}
  let prevTargets = new Map() // id -> {x,y}
  let tracks = new Map() // id -> {id,url,from:{x,y},to:{x,y}}

  // Simple easing
  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
  }
  function lerp(a, b, t) { return a + (b - a) * t }

  let startTs = performance.now()
  let nowTs = startTs
  let rafId
  function tickAnim() {
    nowTs = performance.now()
    // Update last positions without reassigning the map to avoid triggering reactive blocks
    if (Array.isArray(renderItems)) {
      for (const it of renderItems) {
        lastPosMap.set(it.id, { x: it.x, y: it.y })
      }
    }
    // Throttle position target updates (reduce boundary flicker)
    if (hovering) {
      if (nowTs - lastTargetsUpdate > posUpdateMs) {
        posRect = { x0, y0, x1, y1, margin: hoverMargin }
        lastTargetsUpdate = nowTs
      }
    } else {
      // When not hovering, occasionally relax to full view (optional)
      if (nowTs - lastTargetsUpdate > posUpdateMs) {
        posRect = { x0, y0, x1, y1, margin: hoverMargin }
        lastTargetsUpdate = nowTs
      }
    }
    rafId = requestAnimationFrame(tickAnim)
  }
  import { onMount, onDestroy } from 'svelte'
  onMount(() => { rafId = requestAnimationFrame(tickAnim) })
  onDestroy(() => { cancelAnimationFrame(rafId) })

  function posEqual(a, b, eps = 1e-6) {
    if (!a || !b) return false
    return Math.abs(a.x - b.x) <= eps && Math.abs(a.y - b.y) <= eps
  }

  // Recompute targets when items or hover window changes
  $: {
    const nextTargets = new Map()
    const m = new Map()
    let anyChange = false
    for (const it of items) {
      // Explicitly reference hover window vars so this block reacts to them
      const ox = Number(it.x ?? 0); const oy = Number(it.y ?? 0)
      // Use throttled rectangle for position targeting
      const rx0 = posRect?.x0 ?? x0; const ry0 = posRect?.y0 ?? y0
      const rx1 = posRect?.x1 ?? x1; const ry1 = posRect?.y1 ?? y1
      const rmg = posRect?.margin ?? hoverMargin
      const inside = hovering && ox >= (rx0 - rmg) && ox <= (rx1 + rmg) && oy >= (ry0 - rmg) && oy <= (ry1 + rmg)
      let to
      if (inside) {
        // If experimental local packing enabled and we have a packed target, use it
        const lp = localPacked.get(it.id)
        to = (lp && experimentalLocalPacking) ? lp : posFrom(it, 'grid')
      } else {
        to = posFrom(it, 'original')
      }
      const prevTarget = prevTargets.get(it.id)
      if (!posEqual(prevTarget, to)) anyChange = true
      // Animate from last rendered position if available; otherwise from original
      const from = lastPosMap.get(it.id) || posFrom(it, 'original')
      m.set(it.id, { id: it.id, url: it.url, from, to })
      nextTargets.set(it.id, to)
    }
    tracks = m
    // Only reset the animation clock if targets actually changed
    if (anyChange) startTs = performance.now()
    prevTargets = nextTargets
  }

  $: elapsed = Math.max(0, nowTs - startTs)
  $: raw = Math.min(1, duration > 0 ? elapsed / duration : 1)
  $: tNorm = (easing === 'linear') ? raw : easeInOutCubic(raw)

  // Derived layout
  $: cellPx = (gridSize && gridSize > 0) ? Math.min(width, height) / gridSize : 16
  $: imSize = Math.max(minImagePx, Math.floor(cellPx ))
  // Emphasis sizes; smaller square -> larger inside scale
  $: vfRatio = 0.35 / Math.max(0.08, Math.min(1.0, vf))
  
  $: insideScale = Math.max(1.1, 1.15 + 0.3 * (vfRatio - 1))
  // Do not clamp inside size when gridded (allowed to overflow selection)
  $: sizeInside = Math.max(minImagePx, Math.floor(imSize * insideScale))
  $: sizeOutside = Math.max(8, Math.floor(imSize * 0.8))
  // Spacing scale for gridded layout: smaller square => more spacing
  $: spacingScale = Math.max(1.05, 1.1 + 0.1 * (vfRatio - 1))

  // Precompute set of inside ids for rendering (with margin) to avoid per-iteration lookups
  $: insideIds = new Set(items.filter((it) => {
    const ox = Number(it.x ?? 0); const oy = Number(it.y ?? 0)
    return hovering && ox >= (x0 - hoverMargin) && ox <= (x1 + hoverMargin) && oy >= (y0 - hoverMargin) && oy <= (y1 + hoverMargin)
  }).map((it) => it.id))

  // Local packing: compute non-overlapping positions within the throttled rect
  function computeLocalPacked(itemsArr, rect, sizePx, spacingScale, wPx, hPx) {
    if (!rect) return new Map()
    const rx0 = rect.x0, ry0 = rect.y0, rx1 = rect.x1, ry1 = rect.y1
    const margin = rect.margin || 0
    const stepPx = Math.max(1, sizePx * spacingScale)
    const sx = Math.max(1e-6, stepPx / Math.max(1, wPx))
    const sy = Math.max(1e-6, stepPx / Math.max(1, hPx))
    // Items inside rect (with margin)
    const inside = itemsArr.filter((it) => {
      const x = Number(it.x ?? 0), y = Number(it.y ?? 0)
      return x >= (rx0 - margin) && x <= (rx1 + margin) && y >= (ry0 - margin) && y <= (ry1 + margin)
    })
    if (!inside.length) return new Map()
    // Choose center item (closest to rect center)
    const cx = (rx0 + rx1) / 2, cy = (ry0 + ry1) / 2
    let center = inside[0], best = Infinity
    for (const it of inside) {
      const dx = (Number(it.x) - cx), dy = (Number(it.y) - cy)
      const d2 = dx*dx + dy*dy
      if (d2 < best) { best = d2; center = it }
    }
    const baseX = Number(center.x || 0)
    const baseY = Number(center.y || 0)
    function cellToXY(ix, iy) { return { x: baseX + ix * sx, y: baseY + iy * sy } }
    const occ = new Set(); const key = (ix,iy)=> ix+','+iy
    // Reserve center cell at (0,0) so center image stays
    occ.add(key(0,0))
    const ordered = [...inside].sort((a,b)=>{
      if (a.id===center.id) return -1; if (b.id===center.id) return 1
      const da=(a.x-baseX)**2+(a.y-baseY)**2; const db=(b.x-baseX)**2+(b.y-baseY)**2
      return da-db
    })
    const out = new Map(); out.set(center.id, { x: baseX, y: baseY })
    for (const it of ordered) {
      if (it.id === center.id) continue
      const ox = Number(it.x || 0), oy = Number(it.y || 0)
      let ix = Math.round((ox - baseX) / sx)
      let iy = Math.round((oy - baseY) / sy)
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

  $: localPacked = computeLocalPacked(items, posRect, sizeInside, spacingScale, width, height)

  $: renderItems = Array.from(tracks.values()).map((tr) => ({
    id: tr.id,
    url: tr.url,
    x: lerp(tr.from.x, tr.to.x, tNorm),
    y: lerp(tr.from.y, tr.to.y, tNorm),
  }))

  // lastPosMap is updated inside the RAF loop to avoid cycles

</script>

<div
  role="img"
  aria-label="Hover grid minimap"
  class="relative border border-gray-300 bg-white select-none"
  style={`width:${width}px;height:${height}px;`}
  on:mousemove={onMove}
  on:mouseenter={() => (hovering = true)}
  on:mouseleave={() => (hovering = false)}
>
  {#each renderItems as it (it.id)}
    <img
      alt=""
      src={it.url}
      class="absolute object-cover rounded"
      style={`left:${it.x * 100}%;top:${it.y * 100}%;transform:translate(-50%,-50%);width:${(hovering && insideIds.has(it.id) ? sizeInside : sizeOutside)}px;height:${(hovering && insideIds.has(it.id) ? sizeInside : sizeOutside)}px;transition:width 120ms ease,height 120ms ease; z-index:${(hovering && insideIds.has(it.id) ? 10 : 1)}; opacity:${(hovering && insideIds.has(it.id) ? 1 : 0.6)};`}
      loading="lazy"
    />
  {/each}

  <!-- Viewport rectangle -->
  {#if hovering}
    <div
      class="absolute border border-blue-500/70 pointer-events-none"
      style={`left:${x0 * 100}%;top:${y0 * 100}%;width:${viewW * 100}%;height:${viewH * 100}%;`}
    />
  {/if}

  <!-- +/- zoom controls -->
  <div class="absolute top-1 right-1 flex gap-1 bg-white text-white rounded p-0 backdrop-blur-sm">
    <button type="button" class="w-6 h-6 rounded grid place-items-center m-0 bg-black/20 hover:bg-black/30" on:click|stopPropagation={zoomIn} aria-label="Zoom in">+</button>
    <button type="button" class="w-6 h-6 rounded grid place-items-center m-0 bg-black/20 hover:bg-black/30" on:click|stopPropagation={zoomOut} aria-label="Zoom out">−</button>
  </div>
</div>

<style>
</style>
