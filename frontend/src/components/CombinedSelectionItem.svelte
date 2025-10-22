<script>
  import { createEventDispatcher } from 'svelte'
  export let combined // { id, name, posIds:[], negIds:[] }
  export let idToUrl = new Map()
  const dispatch = createEventDispatcher()
  let expanded = false
  function getUrl(id) { return idToUrl instanceof Map ? (idToUrl.get(id) || '') : (idToUrl?.[id] || '') }
  function onDelete() { dispatch('delete', { id: combined.id }) }
</script>

<div class="border rounded bg-white hover:bg-gray-50 p-2 flex flex-col gap-2">
  <div class="flex items-start gap-2">
    <button class="text-left flex-1" on:click={() => (expanded = !expanded)}
            title={`Pos: ${combined.posIds?.length || 0} • Neg: ${combined.negIds?.length || 0}`}>
      <div class="font-medium text-sm truncate">{combined.name || 'Combined selection'}</div>
      <div class="mt-1 text-sm text-gray-600">Pos: {combined.posIds?.length || 0} • Neg: {combined.negIds?.length || 0}</div>
    </button>
    <div class="shrink-0 flex gap-1">
      <button class="px-1 py-0.5 text-sm border rounded" on:click={onDelete} title="Delete">✕</button>
    </div>
  </div>

  {#if expanded}
    <div class="space-y-1">
      <div class="text-sm text-gray-500">Positive</div>
      <div class="flex gap-1">
        {#each (combined.posIds || []).slice(0,5) as gid}
          {#if getUrl(gid)}
            <img src={getUrl(gid)} alt={gid} class="w-16 h-16 object-cover rounded border" />
          {/if}
        {/each}
      </div>
    </div>
    <div class="space-y-1">
      <div class="text-sm text-gray-500">Negative</div>
      <div class="flex gap-1">
        {#each (combined.negIds || []).slice(0,5) as bid}
          {#if getUrl(bid)}
            <img src={getUrl(bid)} alt={bid} class="w-16 h-16 object-cover rounded border" />
          {/if}
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
</style>

