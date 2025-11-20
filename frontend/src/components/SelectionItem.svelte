<script>
  import { createEventDispatcher } from 'svelte'
  export let selection // { id, name, posIds:[], negIds:[], active?:boolean }
  export let idToUrl = new Map()
  const dispatch = createEventDispatcher()
  let expanded = true

  function getUrl(id) { return idToUrl instanceof Map ? (idToUrl.get(id) || '') : (idToUrl?.[id] || '') }
  function onToggle() { dispatch('toggle', { id: selection.id, active: !selection.active }) }
  function onRename() { dispatch('rename', { id: selection.id }) }
  function onDelete() { dispatch('delete', { id: selection.id }) }
</script>

<div class="subtile flex flex-col gap-2 hover:bg-gray-50" draggable="true"
     on:dragstart={(e)=>{ try { const payload = JSON.stringify({ id: selection.id, name: selection.name, posIds: selection.posIds||[], negIds: selection.negIds||[] }); e.dataTransfer.setData('application/x-selection', payload); e.dataTransfer.setData('text/plain', payload); } catch(_) {} }}>
  <div class="flex items-start gap-2">
    <button class="text-left flex-1" on:click={() => (expanded = !expanded)}
            title={`Pos: ${selection.posIds?.length || 0} • Neg: ${selection.negIds?.length || 0}`}>
      <div class="font-medium text-base truncate">{selection.name || 'Selection'}</div>
      <div class="mt-1 text-sm text-gray-600">Pos: {selection.posIds?.length || 0} • Neg: {selection.negIds?.length || 0}</div>
    </button>
    <div class="shrink-0 flex gap-1">
      <button class={`btn btn-xs ${selection.active?'btn-success':''}`} on:click={onToggle} title="Toggle active">{selection.active ? 'On' : 'Off'}</button>
      <button class="btn btn-xs" on:click={onRename} title="Rename">✎</button>
      <button class="btn btn-xs btn-danger" on:click={onDelete} title="Delete">✕</button>
    </div>
  </div>

  {#if expanded}
    <div class="space-y-1">
      <div class="text-sm text-gray-500">Positive examples</div>
      <div class="flex gap-1">
        {#each (selection.posIds || []).slice(0,5) as gid}
          {#if getUrl(gid)}
            <img src={getUrl(gid)} alt={gid} class="w-16 h-16 object-cover rounded border" />
          {/if}
        {/each}
        {#if (selection.posIds?.length || 0) > 5}
          <div class="w-16 h-16 flex items-center justify-center text-sm text-gray-600 border rounded">
            +{(selection.posIds.length - 5)}
          </div>
        {/if}
      </div>
    </div>
    <div class="space-y-1">
      <div class="text-sm text-gray-500">Negative examples</div>
      <div class="flex gap-1">
        {#each (selection.negIds || []).slice(0,5) as bid}
          {#if getUrl(bid)}
            <img src={getUrl(bid)} alt={bid} class="w-16 h-16 object-cover rounded border" />
          {/if}
        {/each}
        {#if (selection.negIds?.length || 0) > 5}
          <div class="w-16 h-16 flex items-center justify-center text-sm text-gray-600 border rounded">
            +{(selection.negIds.length - 5)}
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
</style>
