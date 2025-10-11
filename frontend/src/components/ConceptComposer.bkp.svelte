<script>
  import { createEventDispatcher } from 'svelte'

  // concepts: [{id, name, method, good:[], bad:[] }]
  export let concepts = []
  // allIds: universe of image ids for NOT/complement
  export let allIds = []
  // default method for the new concept (e.g., currentMethodLabel)
  export let defaultMethod = 'avg'

  const dispatch = createEventDispatcher()

  let expr = [] // tokens: {type:'concept', id, negate?:bool} | {type:'op', op:'AND'|'OR'}
  let draftName = ''
  let nextNegate = false

  function addConceptToken(c) {
    expr = [...expr, { type: 'concept', id: c.id, negate: nextNegate }]
    nextNegate = false
  }
  function addOp(op) {
    if (expr.length === 0 || expr[expr.length-1].type === 'op') return
    expr = [...expr, { type: 'op', op }]
  }
  function clearExpr() { expr = []; nextNegate = false }

  function conceptById(id) { return concepts.find(c => c.id === id) }

  function evalSet() {
    if (expr.length === 0) return new Set()
    const universe = new Set(allIds)
    function setFor(token) {
      const c = conceptById(token.id)
      const base = new Set(c?.good || [])
      if (token.negate) {
        const out = new Set()
        for (const id of universe) if (!base.has(id)) out.add(id)
        return out
      }
      return base
    }
    let acc = null
    let pendingOp = null
    for (const t of expr) {
      if (t.type === 'op') { pendingOp = t.op; continue }
      const s = setFor(t)
      if (acc === null) { acc = new Set(s); continue }
      if (pendingOp === 'AND') {
        const next = new Set()
        for (const id of acc) if (s.has(id)) next.add(id)
        acc = next
      } else { // OR
        for (const id of s) acc.add(id)
      }
      pendingOp = null
    }
    return acc || new Set()
  }

  $: preview = Array.from(evalSet())

  function createCombined() {
    const goodIds = Array.from(evalSet())
    if (goodIds.length === 0) return
    const name = draftName && draftName.trim() ? draftName.trim() : expr.map(t => t.type==='op'?t.op:conceptById(t.id)?.name || 'C').join(' ')
    dispatch('create', { name, method: defaultMethod, good: goodIds, bad: [] })
    draftName = ''
    clearExpr()
  }
</script>

<div class="mt-4">
  <div class="text-sm font-medium mb-2">Combine Concepts</div>
  <div class="flex flex-wrap gap-1 mb-2">
    {#each concepts as c (c.id)}
      <button class="px-2 py-0.5 text-xs border rounded hover:bg-gray-50" title={`Good: ${c.good?.length||0} | Bad: ${c.bad?.length||0}`}
              on:click={() => addConceptToken(c)}>{c.name || c.method}</button>
    {/each}
  </div>
  <div class="flex items-center gap-2 mb-2">
    <label class="inline-flex items-center gap-1 text-xs">
      <input type="checkbox" bind:checked={nextNegate} /> NOT next
    </label>
    <button class="px-2 py-0.5 text-xs border rounded" on:click={() => addOp('AND')}>AND</button>
    <button class="px-2 py-0.5 text-xs border rounded" on:click={() => addOp('OR')}>OR</button>
    <button class="px-2 py-0.5 text-xs border rounded" on:click={clearExpr}>Clear</button>
  </div>
  <div class="min-h-[36px] border rounded bg-white px-2 py-1 text-xs flex items-center flex-wrap gap-1">
    {#if expr.length === 0}
      <span class="text-gray-500">Click concepts to add. Use AND/OR and NOT next.</span>
    {:else}
      {#each expr as t, i}
        {#if t.type === 'op'}
          <span class="px-1 py-0.5 rounded bg-amber-100 text-amber-900">{t.op}</span>
        {:else}
          <span class="px-1 py-0.5 rounded bg-blue-100 text-blue-900" title={conceptById(t.id)?.method}>
            {t.negate ? 'NOT ' : ''}{conceptById(t.id)?.name || 'Concept'}
          </span>
        {/if}
      {/each}
    {/if}
  </div>
  <div class="mt-2 text-xs text-gray-600">Preview: {preview.length} images</div>
  <div class="mt-2 flex items-center gap-2">
    <input class="px-2 py-1 border rounded text-sm flex-1 bg-white" placeholder="Name (optional)" bind:value={draftName} />
    <button class="btn" on:click={createCombined} disabled={preview.length===0}>
      <span class="i-heroicons-plus-circle mr-1" /> Create Combined
    </button>
  </div>
</div>

<style>
</style>

