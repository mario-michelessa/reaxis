<script>
  import { createEventDispatcher } from 'svelte'
  // CombinedSelection: show positive (top) and negative (bottom) previews, with apply/rename/delete and collapsible sections
  export let combined // { id, name, posIds: string[], negIds: string[] }
  export let idToUrl = new Map() // Map id -> url
  const dispatch = createEventDispatcher()

  let expandedPos = true
  let expandedNeg = true

  function getUrl(id) { return idToUrl instanceof Map ? (idToUrl.get(id) || '') : (idToUrl?.[id] || '') }
  function onApply() { dispatch('apply', { id: combined?.id, posIds: combined?.posIds || [], negIds: combined?.negIds || [] }) }
  function onRename() {
    const next = prompt('Rename combined selection', combined?.name || 'Combined selection')
    if (next && next.trim()) dispatch('rename', { id: combined?.id, name: next.trim() })
  }
  function onDelete() { dispatch('delete', { id: combined?.id }) }
</script>

<div class="tile">
  <div class="tile-header">
    <div class="truncate flex-1" title={`Pos: ${(combined?.posIds||[]).length} • Neg: ${(combined?.negIds||[]).length}`}>{combined?.name || 'Combined selection'}</div>
    <div class="shrink-0 inline-flex gap-1">
      <button class="btn btn-xs" on:click={onApply} title="Apply to labels">↻</button>
      <button class="btn btn-xs" on:click={onRename} title="Rename">✎</button>
      <button class="btn btn-xs btn-danger" on:click={onDelete} title="Delete">✕</button>
      <button class="btn btn-xs" on:click={() => dispatch('sendToSelections', { selection: { id: `sel:${Date.now()}`, name: combined?.name || 'Selection', posIds: combined?.posIds||[], negIds: combined?.negIds||[], active: true } })} title="Move to selections">⇄</button>
    </div>
  </div>

  <div class="tile-content">
  <!-- Top: Positives -->
  <div class="mb-2">
    <div class="flex items-center justify-between">
      <div class="text-sm text-gray-600">Positive examples ({(combined?.posIds||[]).length})</div>
      <button class="text-sm px-1 py-0.5 border rounded" on:click={() => (expandedPos = !expandedPos)}>{expandedPos ? 'Hide' : 'Show'}</button>
    </div>
    {#if expandedPos}
      <div class="mt-1 flex flex-wrap gap-1" style="max-height: 180px; overflow:auto">
        {#each (combined?.posIds || []).slice(0,200) as id}
          {#if getUrl(id)}
            <img src={getUrl(id)} alt={id} class="w-12 h-12 object-cover rounded border" loading="lazy" />
          {/if}
        {/each}
      </div>
    {/if}
  </div>

  <!-- Bottom: Negatives -->
  <div>
    <div class="flex items-center justify-between">
      <div class="text-sm text-gray-600">Negative examples ({(combined?.negIds||[]).length})</div>
      <button class="text-sm px-1 py-0.5 border rounded" on:click={() => (expandedNeg = !expandedNeg)}>{expandedNeg ? 'Hide' : 'Show'}</button>
    </div>
    {#if expandedNeg}
      <div class="mt-1 flex flex-wrap gap-1" style="max-height: 180px; overflow:auto">
        {#each (combined?.negIds || []).slice(0,200) as id}
          {#if getUrl(id)}
            <img src={getUrl(id)} alt={id} class="w-12 h-12 object-cover rounded border" loading="lazy" />
          {/if}
        {/each}
      </div>
    {/if}
  </div>
  </div>
</div>

<style>
</style>
