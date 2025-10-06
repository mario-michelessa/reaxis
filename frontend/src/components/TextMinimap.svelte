<script>
  // Minimap with a placeable text rectangle that you can position/drag
  // Emits 'confirmRegion' with { rect: {x,y,w,h}, text } in normalized coords
  import AnimatedMinimap from './AnimatedMinimap.svelte'
  import { onMount } from 'svelte'

  // Pass-through props to AnimatedMinimap
  export let items = []
  export let prevItems = []
  export let width = 220
  export let height = 220
  export let gridSize = 0
  export let viewFrac = 0.35
  export let minImagePx = 10
  export let duration = 700
  export let easing = 'easeInOutCubic'

  // Text rectangle state (normalized 0..1)
  export let text = ''
  export let color = '#0ea5e9'
  let placing = false
  let dragging = false
  let dragOff = { x: 0, y: 0 }
  let rect = { x: 0.3, y: 0.3, w: 0.25, h: 0.15 } // default
  let hasRect = false

  let overlay
  let host

  function pxToNorm(px, py) {
    const rectEl = host.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (px - rectEl.left) / rectEl.width))
    const y = Math.max(0, Math.min(1, (py - rectEl.top) / rectEl.height))
    return { x, y }
  }

  function normToPx(nx, ny) {
    return { x: nx * width, y: ny * height }
  }

  function startPlacing() {
    placing = true
    hasRect = false
  }
  // Expose method so parent can trigger placing
  export { startPlacing }

  function onMouseDown(e) {
    if (!placing && !hasRect) return
    const pos = pxToNorm(e.clientX, e.clientY)
    if (placing) {
      rect = { x: pos.x, y: pos.y, w: 0.001, h: 0.001 }
      hasRect = true
      dragging = true
      dragOff = { x: 0, y: 0 }
    } else {
      // Click inside rectangle to drag
      if (pos.x >= rect.x && pos.x <= rect.x + rect.w && pos.y >= rect.y && pos.y <= rect.y + rect.h) {
        dragging = true
        dragOff = { x: pos.x - rect.x, y: pos.y - rect.y }
      }
    }
  }

  function onMouseMove(e) {
    if (!dragging) return
    const pos = pxToNorm(e.clientX, e.clientY)
    if (placing) {
      // update size from anchor (rect.x, rect.y)
      rect.w = Math.max(0.01, Math.abs(pos.x - rect.x))
      rect.h = Math.max(0.01, Math.abs(pos.y - rect.y))
      // place from top-left
      rect.x = Math.min(pos.x, rect.x)
      rect.y = Math.min(pos.y, rect.y)
    } else {
      // dragging: keep size, update origin
      let nx = pos.x - dragOff.x
      let ny = pos.y - dragOff.y
      // clamp to container
      nx = Math.max(0, Math.min(1 - rect.w, nx))
      ny = Math.max(0, Math.min(1 - rect.h, ny))
      rect.x = nx
      rect.y = ny
    }
  }

  function onMouseUp() {
    if (placing) placing = false
    dragging = false
  }

  function confirm() {
    const detail = { rect: { ...rect }, text }
    const ev = new CustomEvent('confirmRegion', { detail })
    host.dispatchEvent(ev)
  }

  function cancelRect() {
    hasRect = false
    placing = false
  }
</script>

<div bind:this={host} class="relative" style={`width:${width}px;height:${height}px;`}>
  <AnimatedMinimap
    {items}
    {prevItems}
    {width}
    {height}
    {gridSize}
    {viewFrac}
    {minImagePx}
    {duration}
    {easing}
  />

  <!-- Overlay interactions -->
  <div class="absolute inset-0" on:mousedown|preventDefault={onMouseDown} on:mousemove|preventDefault={onMouseMove} on:mouseup|preventDefault={onMouseUp} on:mouseleave|preventDefault={onMouseUp} />

  {#if hasRect}
    <div class="absolute border rounded-sm" style={`left:${rect.x*100}%;top:${rect.y*100}%;width:${rect.w*100}%;height:${rect.h*100}%;border-color:${color};background:${color}22;transform:translate(0,0);`} />
    <div class="absolute" style={`left:${rect.x*100}%;top:${rect.y*100 - 4}%;transform:translate(0,-100%);`}>
      <input class="px-1 py-0.5 border rounded bg-white text-sm" placeholder="Enter text" bind:value={text} style="width: 160px;" />
    </div>
  {/if}
</div>

<div class="flex items-center gap-2 mt-2">
  <button class="px-2 py-1 border rounded" on:click={startPlacing}>Add Text Region</button>
  <button class="px-2 py-1 border rounded" on:click={confirm} disabled={!hasRect}>Confirm</button>
  <button class="px-2 py-1 border rounded" on:click={cancelRect} disabled={!hasRect}>Cancel</button>
</div>

<style>
</style>
