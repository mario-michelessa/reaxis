<script>
  export let items = [] // [{ id, url, x, y }] with x,y in [0,1]
  export let width = 200
  export let height = 200
  export let viewFrac = 0.35 // fraction of minimap shown in zoom panel
  export let gridSize = 0 // n_layer from server (snapped grid)
  export let minImagePx = 10 // threshold to switch to dot rendering on minimap
  const overlaySize = 600 // base size of zoom overlay
  // let ovImageSize = Math.floor(viewFrac * overlaySize) // size of images in zoom overlay
  let hovering = false
  let cx = 0.5
  let cy = 0.5

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    cx = Math.min(1, Math.max(0, x))
    cy = Math.min(1, Math.max(0, y))
  }

  function onEnter() {
    hovering = true
  }
  function onLeave() {
    hovering = false
  }

  $: viewW = viewFrac
  $: viewH = viewFrac
  $: x0 = Math.max(0, Math.min(1 - viewW, cx - viewW / 2))
  $: y0 = Math.max(0, Math.min(1 - viewH, cy - viewH / 2))
  $: x1 = x0 + viewW
  $: y1 = y0 + viewH

  // Items within the window, with local coords 0..1
  $: windowItems = items
    .filter((it) => it.x >= x0 && it.x <= x1 && it.y >= y0 && it.y <= y1)
    .map((it) => ({
      ...it,
      lx: (it.x - x0) / viewW,
      ly: (it.y - y0) / viewH,
    }))

  // Scale marker sizes based on grid size and minimap dimensions
  $: cellPx = (gridSize && gridSize > 0) ? Math.min(width, height) / gridSize : 16
  console.log('cellPx', cellPx, 'for gridSize', gridSize)
  $: imSize = Math.max(1, Math.floor(cellPx * 0.95))
  $: showDot = imSize < minImagePx
  // Zoom overlay image size scales with grid size and view fraction
  $: ovImageSize = Math.max(12, Math.floor((cellPx * overlaySize / Math.max(1, width)) * (0.98 / viewFrac)))

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v))
  }
  function zoomIn() {
    viewFrac = clamp(viewFrac * 0.85, 0.08, 0.8)
  }
  function zoomOut() {
    viewFrac = clamp(viewFrac / 0.85, 0.08, 0.8)
  }
</script>

<!-- Minimap container -->
<div
  role="img"
  aria-label="Image scatter minimap"
  class="relative border border-gray-300 bg-white select-none"
  style={`width:${width}px;height:${height}px;`}
  on:mousemove={onMove}
  on:mouseenter={onEnter}
  on:mouseleave={onLeave}
>
  {#each items as it}
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

  <!-- +/- zoom controls -->
  <div class="absolute top-1 right-1 flex gap-1 bg-white text-white rounded p-0 backdrop-blur-sm">
    <button type="button" class="w-6 h-6 rounded grid place-items-center m-0 bg-black/20 hover:bg-black/30" on:click|stopPropagation={zoomIn} aria-label="Zoom in">+</button>
    <button type="button" class="w-6 h-6 rounded grid place-items-center m-0 bg-black/20 hover:bg-black/30" on:click|stopPropagation={zoomOut} aria-label="Zoom out">−</button>
  </div>
</div>

<!-- Zoom overlay centered on screen -->
{#if hovering}
  <div class="fixed inset-0 pointer-events-none grid place-content-center z-50">
    <div class="relative pointer-events-auto rounded shadow-lg border border-gray-200 bg-white p-2">
      <div class="text-sm mb-2 text-gray-700">Zoom</div>
      <div class="relative bg-white" style={`width:${overlaySize}px;height:${overlaySize}px;`}>
        {#each windowItems as it}
          <img
            alt=""
            src={it.url}
            class="absolute object-cover rounded"
            style={`left:${it.lx * 100}%;top:${it.ly * 100}%;transform:translate(-50%,-50%);width:${ovImageSize}px;height:${ovImageSize}px;`}
            loading="lazy"
          />
        {/each}
      </div>
    </div>
  </div>
{/if}

<style>
  /* Ensure crisp overlay without affecting page clicks when hidden */
</style>
