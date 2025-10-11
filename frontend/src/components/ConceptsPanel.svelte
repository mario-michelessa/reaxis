<script>
  import { createEventDispatcher } from 'svelte'

  import ConceptItem from './ConceptItem.svelte'
  export let concepts = [] // [{ id, name, method, good: string[], bad: string[] }]
  // Optional: pass in items [{id, url}] so we can render example thumbnails
  // items: pass full items with {id, url, x, y}
  export let items = []
  const dispatch = createEventDispatcher()

  function selectConcept(c) {
    dispatch('select', { concept: c })
  }

  function deleteConcept(c) {
    dispatch('delete', { id: c.id })
  }

  function renameConcept(c) {
    const next = prompt('Rename concept', c.name || c.method)
    if (next && next.trim() && next !== c.name) {
      dispatch('rename', { id: c.id, name: next.trim() })
    }
  }

  // Note: naming displayed inside ConceptItem

  // Map id -> url for quick lookup in item component
  $: idToUrl = new Map((items || []).map(i => [i.id, i.url]))
</script>

{#if concepts.length === 0}
  <div class="text-xs text-gray-500">No concepts yet. Use "Create concept".</div>
{/if}
<div class="grid gap-2">
  {#each concepts as c (c.id)}
    <ConceptItem
      concept={c}
      {idToUrl}
      on:select={() => selectConcept(c)}
      on:rename={() => renameConcept(c)}
      on:delete={() => deleteConcept(c)}
    />
  {/each}
  
</div>

<style>
</style>
