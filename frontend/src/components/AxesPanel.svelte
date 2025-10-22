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

</script>

<div class="tile">
  <div class="tile-content">
    <!-- Projection radios -->
<div class="mb-2">
  <div class="text-sm text-gray-700 mb-1">Initial projections</div>
  <div class="grid gap-1">
    {#each projections as m}
      <label class="inline-flex items-center gap-2 text-base ml-2">
        <input type="radio" name="axes-proj" value={m} checked={embedSelection===m} on:change={onProjectionChange} />
        <span>{methodAlias(m)}</span>
      </label>
    {/each}
  </div>
  {#if embedSelection === 'dift_sd'}
    <div class="mt-2">
      <div class="text-sm text-gray-700 mb-1">Image-part selection</div>
      <DiftPartSelector selected={diftPart} on:select={(e) => dispatch('selectDiftPart', { part: e.detail.part })} />
    </div>
  {/if}
  </div>

    

{#if count === 0}
  <div class="text-sm text-gray-500">No axes yet. Use Axis Builder or create from concept.</div>
{/if}

<div class="grid gap-2">
  {#each axes as ax (ax.id)}
    <div class="subtile flex items-center justify-between gap-2" draggable={true} on:dragstart={(e)=>onDragStart(e, ax)} title="Drag onto X/Y drop zones">
      <div class="min-w-0 flex-1">
        <div class="text-base font-medium truncate">{ax.name || 'Axis'}</div>
        <div class="text-sm text-gray-500">{Object.keys(ax.coords||{}).length} scores</div>
      </div>
      <div class="shrink-0 inline-flex gap-1">
        <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => dispatch('setX', { id: ax.id })} title="Use as X axis">X</button>
        <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => dispatch('setY', { id: ax.id })} title="Use as Y axis">Y</button>
        <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => onRename(ax)} title="Rename">✎</button>
        <button type="button" class="btn btn-xs btn-danger" on:click={() => onDelete(ax)} title="Delete">✕</button>
      </div>
    </div>
  {/each}
  
</div>
  </div>
</div>

<style>
</style>
