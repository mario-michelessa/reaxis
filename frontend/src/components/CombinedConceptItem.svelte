<script>
  import { createEventDispatcher } from 'svelte'
  export let combined // { id, name, method, good: string[], bad?: string[], expr?: any }
  export let idToUrl = new Map() // Map id -> url
  export let allIds = [] // universe to compute not-retained
  const dispatch = createEventDispatcher()

  let expandedRetained = true
  let expandedNotRetained = false

  function getUrl(id) {
    if (!id) return ''
    return idToUrl instanceof Map ? (idToUrl.get(id) || '') : (idToUrl[id] || '')
  }

  function onLoad() {
    // Send saved chain back to composer
    dispatch('load', { chain: combined?.chain || null })
  }

  $: retained = Array.isArray(combined?.good) ? combined.good : []
  $: universe = new Set(allIds || [])
  $: retainedSet = new Set(retained)
  $: notRetained = Array.from(universe).filter(id => !retainedSet.has(id))
</script>

<div class="border rounded bg-white p-2">
  <div class="flex items-start gap-2">
    <button class="text-left flex-1" on:click={onLoad} title="Load this combined concept into the composer">
      <div class="font-medium text-sm truncate">{combined?.name || 'Combined'}</div>
      <div class="mt-1 text-xs text-gray-600">Retained: {retained.length} • Not retained: {notRetained.length}</div>
    </button>
  </div>

  <div class="mt-2">
    <button class="text-xs px-1 py-0.5 border rounded" on:click={() => expandedRetained = !expandedRetained}>
      {expandedRetained ? 'Hide' : 'Show'} retained examples
    </button>
    {#if expandedRetained}
      <div class="mt-1 flex flex-wrap gap-1">
        {#each retained.slice(0,50) as id}
          {#if getUrl(id)}
            <img src={getUrl(id)} alt={id} class="w-12 h-12 object-cover rounded border" loading="lazy" />
          {/if}
        {/each}
        {#if retained.length > 50}
          <div class="w-12 h-12 flex items-center justify-center text-[11px] text-gray-600 border rounded">+{retained.length - 50}</div>
        {/if}
      </div>
    {/if}
  </div>

  <div class="mt-2">
    <button class="text-xs px-1 py-0.5 border rounded" on:click={() => expandedNotRetained = !expandedNotRetained}>
      {expandedNotRetained ? 'Hide' : 'Show'} not retained examples
    </button>
    {#if expandedNotRetained}
      <div class="mt-1 flex flex-wrap gap-1">
        {#each notRetained.slice(0,50) as id}
          {#if getUrl(id)}
            <img src={getUrl(id)} alt={id} class="w-12 h-12 object-cover rounded border" loading="lazy" />
          {/if}
        {/each}
        {#if notRetained.length > 50}
          <div class="w-12 h-12 flex items-center justify-center text-[11px] text-gray-600 border rounded">+{notRetained.length - 50}</div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
</style>
