<script>
  import { createEventDispatcher } from 'svelte'
  import DiftPartSelector from './DiftPartSelector.svelte'
  import ColorSpaceSelector from './ColorSpaceSelector.svelte'
  // AxesPanel: list and manage 1D axes (scores per image id in [0,1])
  // props:
  // - axes: [{ id, name, coords: Record<string, number> }]
  // - items: optional [{ id, url }] for eventual thumbnails (unused for now)
  // - concepts: optional [{ id, name, method, good: string[], bad: string[] }]
  // - embedSelection: current projection selection; emits 'embedChange' when changed
  // - diftPart: current dift local composition part (e.g., '11')
  export let axes = []
  export let items = []
  export let concepts = []
  export let embedSelection = 'avg'
  export let diftPart = ''
  export let colorSpace = 'color_lch'
  // Labels map for showing counts and creating an axis from lasso selections
  export let labels = new Map()
  // Projections forwarded from App: [{ value, label }]
  export let projections = []
  const dispatch = createEventDispatcher()

  function onDelete(ax) { dispatch('delete', { id: ax.id }) }
  function onRename(ax) {
    const next = prompt('Rename axis', ax.name || 'Axis')
    if (next && next.trim() && next !== ax.name) dispatch('rename', { id: ax.id, name: next.trim() })
  }

  // Drag support: set transferable axis id
  function onDragStart(e, ax) {
    try {
      e.dataTransfer.setData('application/axis-id', ax.id)
      e.dataTransfer.setData('text/plain', ax.id)
      e.dataTransfer.effectAllowed = 'copyMove'
    } catch (_) {}
  }

  $: count = (axes || []).length

  // Projection selection (vertical radios) provided by parent

  function onProjectionChange(e) {
    const val = e.currentTarget?.value
    if (!val) return
    dispatch('embedChange', { selection: val })
  }

  // Helpers for axis creation from labeled examples
  function labelOf(id) {
    try {
      if (!labels) return undefined
      if (labels instanceof Map) return labels.get(id)
      return labels[id]
    } catch (_) { return undefined }
  }
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

  function createAxisFromLabels() {
    // Gather positives/negatives based on current labels
    const goodIds = new Set()
    const badIds = new Set()
    if (labels) {
      if (labels instanceof Map) {
        labels.forEach((v, k) => { if (v === 'good') goodIds.add(k); if (v === 'bad') badIds.add(k) })
      } else {
        for (const [k, v] of Object.entries(labels)) { if (v === 'good') goodIds.add(k); if (v === 'bad') badIds.add(k) }
      }
    }
    const posItems = (items || []).filter(i => goodIds.has(i.id))
    const negItems = (items || []).filter(i => badIds.has(i.id))
    if (posItems.length === 0 && negItems.length === 0) {
      alert('Select positive and negative examples using the lasso tool')
      return
    }
    function dist(a, b) { return Math.hypot((Number(a.x||0) - Number(b.x||0)), (Number(a.y||0) - Number(b.y||0))) }
    function minDist(pt, arr) { if (!arr.length) return Infinity; let m = Infinity; for (const s of arr) { const d = dist(pt, s); if (d < m) m = d } return m }
    const coords = {}
    let maxDp = 0, maxDn = 0
    for (const it of (items || [])) { const dp = minDist(it, posItems); const dn = minDist(it, negItems); if (isFinite(dp) && dp > maxDp) maxDp = dp; if (isFinite(dn) && dn > maxDn) maxDn = dn }
    function clamp01(v) { return Math.max(0, Math.min(1, v)) }
    for (const it of (items || [])) {
      const dp = minDist(it, posItems)
      const dn = minDist(it, negItems)
      let s = 0.5
      if (posItems.length && negItems.length) s = clamp01(dn / (dn + dp + 1e-9))
      else if (posItems.length && !negItems.length) s = clamp01(1 - (dp / (maxDp + 1e-9)))
      else if (negItems.length && !posItems.length) s = clamp01(dn / (maxDn + 1e-9))
      coords[it.id] = s
    }
    const proposed = prompt('Name this axis', 'Axis')
    if (!proposed || !proposed.trim()) return
    const id = `axis:labels:${Date.now()}`
    dispatch('create', { id, name: proposed.trim(), coords })
  }

</script>

<div class="tile">
  <div class="tile-content">
    <div class="text-sm text-gray-700 mb-1">Initial projections</div>
      <div class="grid gap-1">
        {#each projections as opt}
          <label class="inline-flex items-center gap-2 text-base ml-2">
            <input type="radio" name="axes-proj" value={opt.value} checked={embedSelection===opt.value} on:change={onProjectionChange} />
            <span>{opt.label}</span>
          </label>
          {#if embedSelection === 'shape' && opt.value === 'shape'}
            <div class="mt-2 items-center">
              <div class="text-sm text-gray-700 mb-1">Which part?</div>
              <DiftPartSelector selected={diftPart} on:select={(e) => dispatch('selectDiftPart', { part: e.detail.part })} />
            </div>
          {/if}
          {#if embedSelection === 'avg' && opt.value === 'avg'}
            <div class="mt-2 items-center ml-10">
              <!-- <div class="text-sm text-gray-700 mb-1">Color space</div> -->
              <ColorSpaceSelector selected={colorSpace} on:select={(e) => dispatch('selectColorSpace', { space: e.detail.space })} />
            </div>
          {/if}
        {/each}
    </div>
  </div>
</div>
<div class="tile mt-2">
  <div class="tile-content">
  <div class="text-sm text-gray-700 mb-1">Axes</div>

  {#if count === 0}
    <div class="text-sm text-gray-500">No axes yet. Use Axis Builder or create from concept.</div>
  {/if}

<div class="grid gap-2">
  {#each axes as ax (ax.id)}
    <div class="subtile flex items-center justify-between gap-2" draggable={true} on:dragstart={(e)=>onDragStart(e, ax)} title="Drag onto X/Y drop zones">
      <div class="min-w-0 flex-1">
        <div class="text-base font-medium truncate">{ax.name.slice(0, 218) || 'Axis'}</div>
        <!-- <div class="text-sm text-gray-500">{Object.keys(ax.coords||{}).length} scores</div> -->
      </div>
      <div class="shrink-0 inline-flex gap-1 items-center">
        <div class="text-sm text-gray-500">Set: </div>
        <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => dispatch('setX', { id: ax.id })} title="Use as X axis">X</button>
        <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => dispatch('setY', { id: ax.id })} title="Use as Y axis">Y</button>
        <div class="text-sm text-gray-500"> | </div>
        <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => onRename(ax)} title="Rename">✎</button>
        <button type="button" class="btn btn-xs btn-danger" on:click={() => onDelete(ax)} title="Delete">✕</button>
      </div>
    </div>
  {/each}
  <!-- Create new axis tile -->
  <button type="button"
          class="subtile flex items-center justify-between gap-2 text-left"
          on:click={createAxisFromLabels}
          title="Create axis ">
    <div class="min-w-0 flex-1">
      <div class="text-base font-medium truncate flex items-center gap-2">
        <span class="i-heroicons-plus-circle text-slate-600" /> Create new axis
      </div>
      <div class="text-sm text-gray-500 truncate">
        {#if posCount === 0 && negCount === 0}
          Select examples using the lasso tool
        {:else}
          Positive ({posCount}) · Negative ({negCount})
        {/if}
      </div>
    </div>
    <div class="shrink-0 inline-flex gap-1 items-center">
      <span class="btn  btn-xs btn-primary">+</span>
    </div>
  </button>
  
</div>
  </div>
</div>

<style>
</style>
