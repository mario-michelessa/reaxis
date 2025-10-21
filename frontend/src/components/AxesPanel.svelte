<script>
  import { createEventDispatcher } from 'svelte'
  import DiftPartSelector from './DiftPartSelector.svelte'
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

  // Projection selection (vertical radios)
  const projections = ['avg','clip','dino','dift_sd','text']
  function methodAlias(m) {
    if (!m) return ''
    if (m === 'avg') return 'Color'
    if (m === 'clip') return 'Content'
    if (m === 'dino') return 'Global composition'
    if (m === 'dift_sd') return 'Local composition'
    if (m === 'text') return 'Text'
    return m
  }

  function onProjectionChange(e) {
    const val = e.currentTarget?.value
    if (!val) return
    dispatch('embedChange', { selection: val })
  }

  // Create axis from a selected concept using distances (neg -> 0, pos -> 1)
  let selectedConceptId = ''
  function createAxisFromConcept() {
    const concept = (concepts || []).find(c => c.id === selectedConceptId)
    if (!concept) return
    const goodSet = new Set(concept.good || [])
    const badSet = new Set(concept.bad || [])
    const posItems = items.filter(i => goodSet.has(i.id))
    const negItems = items.filter(i => badSet.has(i.id))
    function dist(a,b){ return Math.hypot((Number(a.x||0)-Number(b.x||0)), (Number(a.y||0)-Number(b.y||0))) }
    function minDist(pt, arr){ if (!arr.length) return Infinity; let m=Infinity; for (const s of arr){ const d=dist(pt,s); if (d<m) m=d } return m }
    const coords = {}
    let maxDp = 0, maxDn = 0
    // Precompute maxima for single-class cases
    for (const it of items) {
      const dp = minDist(it, posItems)
      const dn = minDist(it, negItems)
      if (isFinite(dp) && dp>maxDp) maxDp = dp
      if (isFinite(dn) && dn>maxDn) maxDn = dn
    }
    function clamp01(v){ return Math.max(0, Math.min(1, v)) }
    for (const it of items) {
      const dp = minDist(it, posItems)
      const dn = minDist(it, negItems)
      let s = 0.5
      if (posItems.length && negItems.length) {
        // r = 1 baseline; score increases when closer to positives and farther from negatives
        s = clamp01(dn / (dn + dp + 1e-9))
      } else if (posItems.length && !negItems.length) {
        s = clamp01(1 - (dp / (maxDp + 1e-9)))
      } else if (negItems.length && !posItems.length) {
        s = clamp01(dn / (maxDn + 1e-9))
      }
      coords[it.id] = s
    }
    const id = `axis:concept:${concept.id}:${Date.now()}`
    const name = `${(concept.name||concept.method||'Concept')} axis`
    dispatch('create', { id, name, coords })
    selectedConceptId = ''
  }
</script>

<!-- Projection radios -->
<div class="mb-2">
  <div class="text-xs text-gray-700 mb-1">Projection</div>
  <div class="grid gap-1">
    {#each projections as m}
      <label class="inline-flex items-center gap-2 text-sm">
        <input type="radio" name="axes-proj" value={m} checked={embedSelection===m} on:change={onProjectionChange} />
        <span>{methodAlias(m)}</span>
      </label>
    {/each}
  </div>
  {#if embedSelection === 'dift_sd'}
    <div class="mt-2">
      <div class="text-xs text-gray-700 mb-1">Image-part selection</div>
      <DiftPartSelector selected={diftPart} on:select={(e) => dispatch('selectDiftPart', { part: e.detail.part })} />
    </div>
  {/if}
</div>

<!-- Create from concept -->
<div class="mb-3">
  <div class="text-xs text-gray-700 mb-1">Create axis from concept</div>
  <div class="flex items-center gap-2">
    <select class="px-1 py-1 border rounded text-sm flex-1 bg-white" bind:value={selectedConceptId}>
      <option value="">Select a concept…</option>
      {#each concepts as c}
        <option value={c.id}>{c.name || c.method}</option>
      {/each}
    </select>
    <button type="button" class="px-2 py-1 text-xs border rounded" on:click={createAxisFromConcept} disabled={!selectedConceptId}>Create</button>
  </div>
</div>

{#if count === 0}
  <div class="text-xs text-gray-500">No axes yet. Use Axis Builder or create from concept.</div>
{/if}

<div class="grid gap-2">
  {#each axes as ax (ax.id)}
    <div class="border rounded bg-white p-2 flex items-center justify-between gap-2" draggable={true} on:dragstart={(e)=>onDragStart(e, ax)} title="Drag onto X/Y drop zones">
      <div class="min-w-0 flex-1">
        <div class="text-sm font-medium truncate">{ax.name || 'Axis'}</div>
        <div class="text-[11px] text-gray-500">{Object.keys(ax.coords||{}).length} scores</div>
      </div>
      <div class="shrink-0 inline-flex gap-1">
        <button type="button" class="px-2 py-1 text-xs border rounded" on:click={() => dispatch('setX', { id: ax.id })} title="Use as X axis">Set X</button>
        <button type="button" class="px-2 py-1 text-xs border rounded" on:click={() => dispatch('setY', { id: ax.id })} title="Use as Y axis">Set Y</button>
        <button type="button" class="px-2 py-1 text-xs border rounded" on:click={() => onRename(ax)}>Rename</button>
        <button type="button" class="px-2 py-1 text-xs border rounded text-red-600" on:click={() => onDelete(ax)}>Delete</button>
      </div>
    </div>
  {/each}
  
</div>

<style>
</style>
