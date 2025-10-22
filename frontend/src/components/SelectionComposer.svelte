<script>
  import { createEventDispatcher } from 'svelte'
  import SelectionItem from './SelectionItem.svelte'

  export let selections = [] // [{id,name,posIds,negIds,active}]
  export let items = [] // [{id,url}]
  const dispatch = createEventDispatcher()

  $: idToUrl = new Map((items||[]).map(i => [i.id, i.url]))

  function toggleSelection(e) { dispatch('toggle', { id: e.detail.id, active: e.detail.active }) }
  function renameSelection(e) {
    const id = e.detail.id
    const sel = selections.find(s => s.id === id)
    if (!sel) return
    const next = prompt('Rename selection', sel.name || 'Selection')
    if (next && next.trim()) dispatch('rename', { id, name: next.trim() })
  }
  function deleteSelection(e) { dispatch('delete', { id: e.detail.id }) }

  function keepAllPositives() {
    const active = (selections||[]).filter(s => s.active)
    const posSet = new Set()
    for (const s of active) for (const id of (s.posIds||[])) posSet.add(id)
    const posIds = Array.from(posSet)
    const name = prompt('Name positive-union selection', 'Positives union') || 'Positives union'
    dispatch('combine', { id: `combined:${Date.now()}`, name, posIds, negIds: [] })
  }
  function discardAllNegatives() {
    const active = (selections||[]).filter(s => s.active)
    const allIds = new Set((items||[]).map(i => i.id))
    const negSet = new Set()
    for (const s of active) for (const id of (s.negIds||[])) negSet.add(id)
    // Keep all images not labeled as negative anywhere
    const posIds = Array.from(allIds).filter(id => !negSet.has(id))
    const name = prompt('Name not-negatives selection', 'Not negatives') || 'Not negatives'
    dispatch('combine', { id: `combined:${Date.now()}`, name, posIds, negIds: [] })
  }
</script>

<div class="mt-4 tile tile-primary">
  <div class="tile-content">
    <div class="text-base font-semibold mb-2">Selections</div>
  <div class="grid gap-2">
    {#each selections as s (s.id)}
      <SelectionItem selection={s}
                     {idToUrl}
                     on:toggle={toggleSelection}
                     on:rename={renameSelection}
                     on:delete={deleteSelection} />
    {/each}
    {#if (selections||[]).length === 0}
      <div class="text-sm text-gray-500">No selections yet. Use “Save selection” near the minimap.</div>
    {/if}
  </div>

  <div class="mt-3 flex gap-2">
    <button class="btn btn-success btn-sm disabled:opacity-50" on:click={keepAllPositives} disabled={(selections||[]).filter(s => s.active).length===0}>
      Keep all positives
    </button>
    <button class="btn btn-amber btn-sm disabled:opacity-50" on:click={discardAllNegatives} disabled={(selections||[]).filter(s => s.active).length===0}>
      Discard all negatives
    </button>
  </div>
  </div>
</div>

<style>
</style>
