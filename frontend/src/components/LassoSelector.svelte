<script>
  import { createEventDispatcher, onMount, onDestroy } from 'svelte'
  // LassoSelector: freehand polygon selection overlay
  // props:
  // - width, height: pixel size of overlay
  // - enabled: boolean; when true, listens to pointer events
  // - items: [{ id, x, y }] with x,y in [0,1] (normalized coordinates)
  // Emits 'select' with { ids, polygon } on pointer up when a path exists
  export let width = 600
  export let height = 600
  export let enabled = false
  export let items = []
  // Sampling controls (larger = fewer points, smoother)
  export let sampleDistPx = 4
  export let sampleMs = 20
  // Visuals
  export let strokeColor = '#3b82f6' // blue-500
  export let fillColor = 'rgba(59,130,246,0.12)'
  const dispatch = createEventDispatcher()

  let container
  let drawing = false
  let points = [] // pixel coords [{x,y}]
  let lastPoint = null
  let lastAddTs = 0

  function toLocal(e) {
    const rect = container.getBoundingClientRect()
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left))
    const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top))
    return { x, y }
  }

  function onPointerDown(e) {
    if (!enabled) return
    e.preventDefault()
    e.stopPropagation()
    drawing = true
    const p = toLocal(e)
    points = [p]
    lastPoint = p
    lastAddTs = performance.now()
    try { container.setPointerCapture(e.pointerId) } catch(_) {}
  }
  function onPointerMove(e) {
    if (!enabled || !drawing) return
    e.preventDefault()
    e.stopPropagation()
    const p = toLocal(e)
    const now = performance.now()
    const dx = p.x - (lastPoint?.x ?? p.x)
    const dy = p.y - (lastPoint?.y ?? p.y)
    const d2 = dx*dx + dy*dy
    if ((now - lastAddTs) >= sampleMs && d2 >= (sampleDistPx*sampleDistPx)) {
      points = [...points, p]
      lastPoint = p
      lastAddTs = now
    }
  }
  function onPointerUp(e) {
    if (!enabled) return
    e.preventDefault()
    e.stopPropagation()
    if (drawing) {
      drawing = false
      finalizeSelection()
    }
    try { container.releasePointerCapture(e.pointerId) } catch(_) {}
  }

  function pointInPoly(px, py, poly) {
    let inside = false
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i].x, yi = poly[i].y
      const xj = poly[j].x, yj = poly[j].y
      const intersect = ((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / ((yj - yi) || 1e-9) + xi)
      if (intersect) inside = !inside
    }
    return inside
  }

  function finalizeSelection() {
    if (!Array.isArray(points) || points.length < 3) { points = []; return }
    const norm = points.map(p => ({ x: p.x / Math.max(1, width), y: p.y / Math.max(1, height) }))
    const ids = []
    for (const it of items || []) {
      const px = Math.max(0, Math.min(1, Number(it.x || 0)))
      const py = Math.max(0, Math.min(1, Number(it.y || 0)))
      if (pointInPoly(px, py, norm)) ids.push(it.id)
    }
    dispatch('select', { ids, polygon: norm })
  }

  function clear() { points = [] }
  export function reset() { points = []; drawing = false }
</script>

<div
  bind:this={container}
  class="absolute inset-0"
  style={`width:${width}px;height:${height}px;`}
  on:pointerdown={onPointerDown}
  on:pointermove={onPointerMove}
  on:pointerup={onPointerUp}
  on:pointercancel={onPointerUp}
  on:mouseleave={onPointerUp}
  aria-hidden={!enabled}
>
  {#if enabled && points.length > 0}
    <svg class="absolute inset-0" width={width} height={height} viewBox={`0 0 ${width} ${height}`}
         style="pointer-events:none">
      <polyline
        fill={fillColor}
        stroke={strokeColor}
        stroke-width="2"
        points={points.map(p => `${p.x},${p.y}`).join(' ')}
      />
    </svg>
  {/if}
</div>

<style>
</style>
