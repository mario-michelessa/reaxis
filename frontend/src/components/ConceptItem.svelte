<script>
  import { createEventDispatcher } from 'svelte'

  export let concept
  // Map or object: id -> url
  export let idToUrl = new Map()

  const dispatch = createEventDispatcher()
  let expanded = true

  function methodAlias(m) {
    if (!m) return ''
    if (m === 'avg') return 'Color (avg)'
    if (m === 'clip') return 'Content (clip)'
    if (m === 'dino') return 'Global composition (dino)'
    if (m === 'dift_sd') return 'Local composition (dift_sd)'
    if (m.startsWith('dift_sd_part')) return `Local composition (dift_sd) ${m.replace('dift_sd_part','part ')}`
    return m
  }

  function displayName(c) {
    if (c?.name && (c.name === c.method)) return methodAlias(c.method)
    return c?.name || methodAlias(c?.method)
  }

  function getUrl(id) {
    if (!id) return ''
    // Support both Map and plain object
    return idToUrl instanceof Map ? (idToUrl.get(id) || '') : (idToUrl[id] || '')
  }

  function onLoadConcept() {
    dispatch('select', { concept })
  }

  function onRename() {
    dispatch('rename', { id: concept.id })
  }

  function onDelete() {
    dispatch('delete', { id: concept.id })
  }
</script>

<div class="border rounded bg-white hover:bg-gray-50 p-2 flex flex-col gap-2">
  <div class="flex items-start gap-2">
    <button class="text-left flex-1" on:click={() => (expanded = !expanded)}
            title={`Method: ${methodAlias(concept.method)} | Good: ${concept.good?.length || 0} | Bad: ${concept.bad?.length || 0}`}>
      <div class="font-medium text-sm truncate">{displayName(concept)}</div>
      <div class="mt-1 text-xs text-gray-600">Good: {concept.good?.length || 0} • Bad: {concept.bad?.length || 0}</div>
    </button>
    <div class="shrink-0 flex gap-1">
      <button class="px-1 py-0.5 text-xs border rounded" on:click={onLoadConcept} title="Load">↻</button>
      <button class="px-1 py-0.5 text-xs border rounded" on:click={onRename} title="Rename">✎</button>
      <button class="px-1 py-0.5 text-xs border rounded" on:click={onDelete} title="Delete">✕</button>
    </div>
  </div>

  {#if expanded}
    <div class="space-y-1">
      <div class="text-xs text-gray-500">Positive examples</div>
      <div class="flex gap-1">
        {#each (concept.good || []).slice(0,5) as gid}
          {#if getUrl(gid)}
            <img src={getUrl(gid)} alt={gid} class="w-16 h-16 object-cover rounded border" />
          {/if}
        {/each}
        {#if (concept.good?.length || 0) > 5}
          <div class="w-16 h-16 flex items-center justify-center text-xs text-gray-600 border rounded">
            +{(concept.good.length - 5)}
          </div>
        {/if}
      </div>
    </div>
    <div class="space-y-1">
      <div class="text-xs text-gray-500">Negative examples</div>
      <div class="flex gap-1">
        {#each (concept.bad || []).slice(0,5) as bid}
          {#if getUrl(bid)}
            <img src={getUrl(bid)} alt={bid} class="w-16 h-16 object-cover rounded border" />
          {/if}
        {/each}
        {#if (concept.bad?.length || 0) > 5}
          <div class="w-16 h-16 flex items-center justify-center text-xs text-gray-600 border rounded">
            +{(concept.bad.length - 5)}
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
</style>
