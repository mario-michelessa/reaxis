<script>
  // Animated minimap in Svelte with transitions between previous and current positions
  export let items = [] // [{ id, url, gx, gy, x?, y? }]
  export let prevItems = [] // previous embedding state (same shape)
  export let width = 200
  export let height = 200
  export let viewFrac = 0.35
  export let gridSize = 0
  export let minImagePx = 10
  export let duration = 700
  export let easing = 'easeInOutCubic'
  // Inline zoom panel below the minimap
  export let zoomPanel = true
  export let zoomWidth = 700
  export let zoomHeight = 700

  let hovering = false
  let cx = 0.5
  let cy = 0.5
  let vf = viewFrac

  $: vf = viewFrac

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

  // Scribble overlay (optional)
  import { createEventDispatcher } from 'svelte'
  const dispatch = createEventDispatcher()
  export let enableScribble = false
  export let brushRadiusPx = 18
  export let activeLabel = 'good' // 'good' | 'bad'
  export let colors = { good: '#16a34a', bad: '#dc2626' }
  export let labels = new Map() // id -> 'good' | 'bad'
  export let labelStrategy = (center, items, radiusPx, dims) => {
    const r2 = radiusPx * radiusPx
    const out = []
    for (const it of items) {
      const ix = it.x * dims.width
      const iy = it.y * dims.height
      const dx = ix - center.xPx
      const dy = iy - center.yPx
      if (dx * dx + dy * dy <= r2) out.push(it.id)
    }
    return out
  }
  let overlay
  let octx
  let drawing = false
  let lastPt = null
  function setLabel(ids, label) {
    const next = new Map(labels)
    for (const id of ids) next.set(id, label)
    labels = next
    dispatch('label', { ids, label, labels })
  }
  function getPointer(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const xPx = e.clientX - rect.left
    const yPx = e.clientY - rect.top
    return { xPx, yPx, x: Math.max(0, Math.min(1, xPx / rect.width)), y: Math.max(0, Math.min(1, yPx / rect.height)) }
  }
  function beginScribble(e) {
    if (!enableScribble) return
    drawing = true
    lastPt = getPointer(e)
    drawDot(lastPt)
    applyBrush(lastPt)
  }
  function moveScribble(e) {
    if (!enableScribble || !drawing) return
    const p = getPointer(e)
    drawStroke(lastPt, p)
    applyBrush(p)
    lastPt = p
  }
  function endScribble() {
    drawing = false
    lastPt = null
  }
  function drawDot(p) {
    if (!octx) return
    octx.beginPath()
    octx.arc(p.xPx, p.yPx, brushRadiusPx / 2, 0, Math.PI * 2)
    octx.fillStyle = (activeLabel === 'good' ? colors.good : colors.bad) + '55'
    octx.fill()
  }
  function drawStroke(a, b) {
    if (!octx || !a || !b) return
    octx.strokeStyle = (activeLabel === 'good' ? colors.good : colors.bad) + '88'
    octx.lineWidth = brushRadiusPx
    octx.lineCap = 'round'
    octx.beginPath()
    octx.moveTo(a.xPx, a.yPx)
    octx.lineTo(b.xPx, b.yPx)
    octx.stroke()
  }
  function applyBrush(p) {
    const ids = labelStrategy(p, renderItems, brushRadiusPx, { width, height })
    if (ids && ids.length) setLabel(ids, activeLabel)
  }
  export function clearScribble() { if (octx) octx.clearRect(0, 0, width, height) }

  function posFrom(it, which) {
    if (!it) return { x: 0, y: 0 }
    if (which === 'grid') return { x: Number(it.gx ?? it.x ?? 0), y: Number(it.gy ?? it.y ?? 0) }
    return { x: Number(it.x ?? 0), y: Number(it.y ?? 0) }
  }

  // Build track map on every items/prevItems change
  let tracks = new Map()
  $: {
    const prevMap = new Map(prevItems.map((p) => [p.id, p]))
    const m = new Map()
    for (const it of items) {
      const prev = prevMap.get(it.id)
      // Use original (non-grid) positions for the animated minimap
      const from = posFrom(prev || it, 'original')
      const to = posFrom(it, 'original')
      m.set(it.id, { id: it.id, url: it.url, from, to })
    }
    tracks = m
    startTs = performance.now()
  }

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
    rafId = requestAnimationFrame(tickAnim)
  }

  import { onMount, onDestroy } from 'svelte'
  onMount(() => { rafId = requestAnimationFrame(tickAnim) })
  onDestroy(() => { cancelAnimationFrame(rafId) })

  $: elapsed = Math.max(0, nowTs - startTs)
  $: raw = Math.min(1, duration > 0 ? elapsed / duration : 1)
  $: tNorm = (easing === 'linear') ? raw : easeInOutCubic(raw)

  // Derived layout and window
  $: cellPx = (gridSize && gridSize > 0) ? Math.min(width, height) / gridSize : 16
  $: imSize = Math.max(1, Math.floor(cellPx * 0.95))
  $: showDot = imSize < minImagePx

  $: viewW = vf
  $: viewH = vf
  $: x0 = Math.max(0, Math.min(1 - viewW, cx - viewW / 2))
  $: y0 = Math.max(0, Math.min(1 - viewH, cy - viewH / 2))
  $: x1 = x0 + viewW
  $: y1 = y0 + viewH

  $: renderItems = Array.from(tracks.values()).map((tr) => ({
    id: tr.id,
    url: tr.url,
    x: lerp(tr.from.x, tr.to.x, tNorm),
    y: lerp(tr.from.y, tr.to.y, tNorm)
  }))

  // Filtered window items for minimap (original coords)
  $: windowItems = renderItems
    .filter((it) => it.x >= x0 && it.x <= x1 && it.y >= y0 && it.y <= y1)
    .map((it) => ({ ...it, lx: (it.x - x0) / viewW, ly: (it.y - y0) / viewH }))

  // For the zoom overlay panel, use grid (snapped) coordinates
  $: gridItems = (items || []).map((it) => ({ id: it.id, url: it.url, x: posFrom(it, 'grid').x, y: posFrom(it, 'grid').y }))
  $: windowGridItems = gridItems
    .filter((it) => it.x >= x0 && it.x <= x1 && it.y >= y0 && it.y <= y1)
    .map((it) => ({ ...it, lx: (it.x - x0) / viewW, ly: (it.y - y0) / viewH }))
</script>

<!-- Minimap container -->
<div
  role="img"
  aria-label="Animated image scatter minimap"
  class="relative border border-gray-300 bg-white select-none"
  style={`width:${width}px;height:${height}px;`}
  on:mousemove={onMove}
  on:mouseenter={() => (hovering = true)}
  on:mouseleave={() => (hovering = false)}
>
  {#each renderItems as it (it.id)}
    {#if showDot}
      <div
        class="absolute rounded-full bg-gray-700 border border-white/70"
        style={`left:${it.x * 100}%;top:${it.y * 100}%;transform:translate(-50%,-50%);width:6px;height:6px;`}
        aria-hidden="true"
      />
    {:else}
      <img
        alt=""
        src={it.url}
        class="absolute object-cover rounded"
        style={`left:${it.x * 100}%;top:${it.y * 100}%;transform:translate(-50%,-50%);width:${imSize}px;height:${imSize}px;`}
        loading="lazy"
      />
    {/if}
  {/each}

  <!-- Viewport rectangle -->
  <div
    class="absolute border border-blue-500/70 pointer-events-none"
    style={`left:${x0 * 100}%;top:${y0 * 100}%;width:${viewW * 100}%;height:${viewH * 100}%;`}
  />

  {#if enableScribble}
    <!-- Labeled overlays: green/red translucent squares on top of items -->
    {#each renderItems as it (it.id)}
      {#if labels.get(it.id)}
        <div
          class="absolute rounded-sm"
          style={`left:${it.x * 100}%;top:${it.y * 100}%;transform:translate(-50%,-50%);width:${Math.max(8, imSize)}px;height:${Math.max(8, imSize)}px;background:${labels.get(it.id)==='good'?colors.good:colors.bad};opacity:0.25;pointer-events:none;`}
          aria-hidden="true"
        />
      {/if}
    {/each}
  {/if}

  <!-- +/- zoom controls -->
  <div class="absolute top-1 right-1 flex gap-1 bg-white text-white rounded p-0 backdrop-blur-sm">
    <button type="button" class="w-6 h-6 rounded grid place-items-center m-0 bg-black/20 hover:bg-black/30" on:click|stopPropagation={zoomIn} aria-label="Zoom in">+</button>
    <button type="button" class="w-6 h-6 rounded grid place-items-center m-0 bg-black/20 hover:bg-black/30" on:click|stopPropagation={zoomOut} aria-label="Zoom out">−</button>
  </div>

  {#if enableScribble}
    <canvas bind:this={overlay}
            width={width}
            height={height}
            class="absolute inset-0"
            on:mousedown|preventDefault={beginScribble}
            on:mousemove|preventDefault={moveScribble}
            on:mouseup|preventDefault={endScribble}
            on:mouseleave|preventDefault={endScribble}
            on:click|stopPropagation
    />
  {/if}
</div>
<!-- 
{#if zoomPanel}
  <div class="mt-2">
    <div class="text-sm mb-1 text-gray-700">Zoom</div>
    <div class="relative bg-white border border-gray-200" style={`width:${zoomWidth}px;height:${zoomHeight}px;`}>
      {#each windowGridItems as it (it.id)}
        <img
          alt=""
          src={it.url}
          class="absolute object-cover rounded"
          style={`left:${it.lx * 100}%;top:${it.ly * 100}%;transform:translate(-50%,-50%);width:${Math.max(12, Math.floor((cellPx * zoomWidth / Math.max(1, width)) * (0.98 / viewW)))}px;height:${Math.max(12, Math.floor((cellPx * zoomWidth / Math.max(1, width)) * (0.98 / viewW)))}px;`}
          loading="lazy"
        />
      {/each}
    </div>
  </div>
{/if} -->

<style>
</style>
