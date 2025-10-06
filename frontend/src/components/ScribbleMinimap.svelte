<script>
  import { createEventDispatcher, onMount } from 'svelte'

  // Props
  export let items = [] // [{ id, url, x, y }] with x,y in [0,1]
  export let width = 220
  export let height = 220
  export let brushRadiusPx = 18
  // active label: 'good' or 'bad'
  export let active = 'good'
  export let colors = { good: '#16a34a', bad: '#dc2626' }
  // External labeling strategy (modular). Default: label items within radiusPx in pixel space
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

  // Labeled state
  export let labels = new Map() // id -> 'good' | 'bad'
  const dispatch = createEventDispatcher()

  let container
  let canvas
  let ctx
  let drawing = false
  let last = null

  function setLabel(ids, label) {
    // update labels Map (replace to trigger reactivity)
    const next = new Map(labels)
    for (const id of ids) next.set(id, label)
    labels = next
    dispatch('label', { ids, label, labels })
  }

  function getPointer(e) {
    const rect = container.getBoundingClientRect()
    const xPx = e.clientX - rect.left
    const yPx = e.clientY - rect.top
    return {
      xPx,
      yPx,
      x: Math.max(0, Math.min(1, xPx / rect.width)),
      y: Math.max(0, Math.min(1, yPx / rect.height)),
    }
  }

  function beginDraw(e) {
    drawing = true
    last = getPointer(e)
    drawDot(last)
    applyBrush(last)
  }

  function moveDraw(e) {
    if (!drawing) return
    const p = getPointer(e)
    drawStroke(last, p)
    applyBrush(p)
    last = p
  }

  function endDraw() {
    drawing = false
    last = null
  }

  function drawDot(p) {
    if (!ctx) return
    ctx.beginPath()
    ctx.arc(p.xPx, p.yPx, brushRadiusPx / 2, 0, Math.PI * 2)
    ctx.fillStyle = (active === 'good' ? colors.good : colors.bad) + '55'
    ctx.fill()
  }

  function drawStroke(a, b) {
    if (!ctx || !a || !b) return
    ctx.strokeStyle = (active === 'good' ? colors.good : colors.bad) + '88'
    ctx.lineWidth = brushRadiusPx
    ctx.lineCap = 'round'
    ctx.beginPath()
    ctx.moveTo(a.xPx, a.yPx)
    ctx.lineTo(b.xPx, b.yPx)
    ctx.stroke()
  }

  function applyBrush(p) {
    const ids = labelStrategy(p, items, brushRadiusPx, { width, height })
    if (ids && ids.length) setLabel(ids, active)
  }

  function clearScribble() {
    if (ctx) ctx.clearRect(0, 0, width, height)
  }

  // Expose methods for parent if needed
  export { clearScribble, setLabel }

  onMount(() => {
    ctx = canvas.getContext('2d')
  })
</script>

<div bind:this={container}
     class="relative select-none"
     style={`width:${width}px;height:${height}px;`}
     on:mousedown|preventDefault={beginDraw}
     on:mousemove|preventDefault={moveDraw}
     on:mouseup|preventDefault={endDraw}
     on:mouseleave|preventDefault={endDraw}
>
  <!-- Minimap points/images -->
  {#each items as it (it.id)}
    <div
      class="absolute rounded-full border"
      style={`left:${it.x*100}%;top:${it.y*100}%;transform:translate(-50%,-50%);width:6px;height:6px;background:${labels.get(it.id)==='good'?colors.good:labels.get(it.id)==='bad'?colors.bad:'#6b7280'};border-color:#fff7;`}
      title={it.id}
    />
  {/each}

  <!-- Scribble canvas overlay -->
  <canvas bind:this={canvas}
          width={width}
          height={height}
          class="absolute inset-0 pointer-events-none"
  />
</div>

<!-- Lightweight toolbar (optional) -->
<div class="flex items-center gap-2 mt-2">
  <div class="text-sm">Brush:</div>
  <button class="px-2 py-1 rounded text-white" style={`background:${colors.good}`} on:click={() => active='good'} aria-label="Good">Good</button>
  <button class="px-2 py-1 rounded text-white" style={`background:${colors.bad}`} on:click={() => active='bad'} aria-label="Bad">Bad</button>
  <label class="text-sm ml-2">Radius
    <input type="range" min="4" max="60" step="1" bind:value={brushRadiusPx} class="align-middle ml-1" />
  </label>
  <button class="px-2 py-1 border rounded ml-2" on:click={clearScribble}>Clear Scribble</button>
</div>

<style>
</style>

