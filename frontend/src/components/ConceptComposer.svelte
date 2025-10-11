<script>
  import { createEventDispatcher } from 'svelte'
  import ConceptTolerance from './ConceptTolerance.svelte'

  // concepts: [{id, name, method, good:[], bad:[] }]
  export let concepts = []
  // allIds: universe of image ids for NOT/complement
  export let allIds = []
  // default method for the new concept (e.g., currentMethodLabel)
  export let defaultMethod = 'avg'
  // items2d: [{ id, url, x, y }] for current embedding view
  export let items2d = []
  // API base and dataset path for per-concept projection fetching
  export let apiBase = ''
  export let datasetPath = ''

  const dispatch = createEventDispatcher()

  let combinedName = ''

  // Build labeled items per concept for tolerance visualization
  function labeledItemsForConcept(c) {
    if (!c) return []
    const good = new Set(c.good || [])
    const bad = new Set(c.bad || [])
    const src = (methodCache.get(c.method) || items2d || [])
    try { console.log('[ConceptComposer] labeledItemsForConcept', { id: c.id, method: c.method, srcLen: src.length, good: good.size, bad: bad.size }) } catch (_) {}
    return src.map((it) => ({
      id: it.id,
      url: it.url,
      x: it.x,
      y: it.y,
      label: good.has(it.id) ? 'pos' : (bad.has(it.id) ? 'neg' : undefined),
    }))
  }

  // Per-concept projection cache: method -> [{id,url,x,y}]
  let methodCache = new Map()
  let loading = new Set()
  function prefixUrl(u) {
    if (!u) return ''
    if (u.startsWith('http://') || u.startsWith('https://')) return u
    if (u.startsWith('/')) return (apiBase || '') + u
    return u
  }
  async function fetchItemsForMethod(method) {
    if (!method || methodCache.has(method) || loading.has(method)) return
    loading.add(method)
    try {
      try { console.log('[ConceptComposer] fetch start', { method }) } catch(_) {}
      const qs = new URLSearchParams()
      if (datasetPath && datasetPath.trim()) qs.set('dataset', datasetPath.trim())
      qs.set('embed', method)
      qs.set('method', 'pca')
      const url = `${apiBase}/gallery.json?${qs.toString()}`
      const res = await fetch(url)
      if (!res.ok) throw new Error('fetch failed')
      const data = await res.json()
      const items = (data.items || []).map((it) => ({
        id: it.id,
        url: prefixUrl(it.url || it.path),
        x: Number(it.x ?? it.gx ?? 0),
        y: Number(it.y ?? it.gy ?? 0),
      }))
      methodCache.set(method, items)
      try { console.log('[ConceptComposer] fetch done', { method, items: items.length }) } catch(_) {}
    } catch (e) {
      console.warn('[ConceptComposer] fetchItemsForMethod error', e)
    } finally {
      loading.delete(method)
    }
  }
  // Prefetch projections for all concepts' methods
  $: (function prefetchConceptMethods(methods) {
    const uniq = Array.from(new Set(methods))
    try { console.log('[ConceptComposer] prefetchConceptMethods', uniq) } catch(_) {}
    for (const m of uniq) fetchItemsForMethod(m)
  })(concepts.map(c => c.method))

  // Reset cache if dataset changes
  $: (function onDatasetChange(_) {
    methodCache = new Map(); loading = new Set()
  })(datasetPath)

  // Boolean ops and inferred tracking
  let ops = [] // ops[i] is op between concepts[i] and concepts[i+1]
  $: if (ops.length !== Math.max(0, concepts.length - 1)) {
    ops = Array.from({ length: Math.max(0, concepts.length - 1) }, (_, i) => ops[i] || 'AND')
  }
  let inferredById = new Map()
  // Per-concept tolerance and flags (e.g., p from slider)
  let tolById = new Map() // conceptId -> { p: number, active?: boolean }
  function arraysEqual(a = [], b = []) {
    if (a === b) return true
    if (!Array.isArray(a) || !Array.isArray(b)) return false
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
    return true
  }
  function onInferred(conceptId, detail) {
    try { console.log('[ConceptComposer] onInferred', { conceptId, pos: (detail.posIds||[]).length, neg: (detail.negIds||[]).length }) } catch(_) {}
    const prev = inferredById.get(conceptId) || { posIds: [], negIds: [] }
    const nextVal = { posIds: detail.posIds || [], negIds: detail.negIds || [] }
    if (arraysEqual(prev.posIds, nextVal.posIds) && arraysEqual(prev.negIds, nextVal.negIds)) {
      try { console.log('[ConceptComposer] onInferred no-change', { conceptId }) } catch(_) {}
      // Still update tolerance if provided and changed
      const prevTol = tolById.get(conceptId) || { p: 50, active: true }
      const nextTol = { p: Number(detail.p ?? prevTol.p ?? 50), active: Boolean(detail.active ?? prevTol.active ?? true) }
      if (prevTol.p !== nextTol.p || prevTol.active !== nextTol.active) {
        tolById = new Map(tolById).set(conceptId, nextTol)
      }
      return
    }
    const next = new Map(inferredById)
    next.set(conceptId, nextVal)
    inferredById = next
    // Track tolerance settings as well
    const prevTol = tolById.get(conceptId) || { p: 50, active: true }
    const nextTol = { p: Number(detail.p ?? prevTol.p ?? 50), active: Boolean(detail.active ?? prevTol.active ?? true) }
    if (prevTol.p !== nextTol.p || prevTol.active !== nextTol.active) {
      tolById = new Map(tolById).set(conceptId, nextTol)
    }
  }

  // Debug current op and highlight sizes per concept
  $: (function debugChain() {
    try {
      if ((concepts || []).length >= 1) {
        const info = concepts.map((c, i) => {
          const prevId = i>0 ? concepts[i-1].id : null
          const op = i>0 ? ops[i-1] : null
          const prevInf = prevId ? inferredById.get(prevId) : null
          const cpos = (i>0 && op==='OR' && prevInf) ? (prevInf.posIds||[]).length : 0
          const cneg = (i>0 && op==='AND' && prevInf) ? (prevInf.negIds||[]).length : 0
          return { i, id: c.id, op, cpos, cneg }
        })
        console.log('[ConceptComposer] chain', info)
      }
    } catch(_) {}
  })()

  // Allow parent to load a saved chain (ops between concepts)
  export let loadChain = null // { conceptIds: string[], ops: string[], tolerances?: {id:string,p:number,active?:boolean}[] }
  $: if (loadChain && Array.isArray(loadChain.ops)) {
    try { console.log('[ConceptComposer] loadChain', loadChain) } catch(_) {}
    const want = loadChain.ops || []
    const n = Math.max(0, concepts.length - 1)
    ops = Array.from({ length: n }, (_, i) => want[i] || 'AND')
    // Load tolerances per concept if provided
    if (Array.isArray(loadChain.tolerances)) {
      const m = new Map()
      for (const t of loadChain.tolerances) {
        if (t && t.id) m.set(t.id, { p: Number(t.p ?? 50), active: Boolean(t.active ?? true) })
      }
      tolById = m
    }
  }

  // Compute combined good ids from chain: start with concept[0] inferred pos, apply AND/OR with next concepts' inferred pos
  function getInferredPosFrom(map, cId) {
    const v = map.get(cId)
    return Array.isArray(v?.posIds) ? v.posIds : []
  }
  function setIntersect(a, b) {
    const sb = new Set(b)
    return a.filter(x => sb.has(x))
  }
  function setUnion(a, b) {
    const s = new Set(a)
    for (const x of b) s.add(x)
    return Array.from(s)
  }
  $: combinedGood = (function computeCombined(depInferred) {
    if (!concepts || concepts.length === 0) return []
    let acc = [...getInferredPosFrom(depInferred, concepts[0].id)]
    for (let i = 1; i < concepts.length; i++) {
      const op = ops[i-1]
      const cur = getInferredPosFrom(depInferred, concepts[i].id)
      acc = (op === 'AND') ? setIntersect(acc, cur) : setUnion(acc, cur)
    }
    try { console.log('[ConceptComposer] combinedGood', acc.length) } catch(_) {}
    return acc
  })(inferredById)
</script>

<!-- Old builder UI removed as requested -->

<!-- Per-concept tolerance views with boolean ribbons -->
{#if concepts.length > 0}
  {#key datasetPath}
  <div class="mt-6">
    <div class="text-sm font-medium mb-2">Concept Tolerance</div>
    <div class="grid gap-3">
      {#each concepts as c, i (c.id)}
        {#if i > 0}
          <div class="flex items-center justify-center gap-2 text-xs">
            <span class="text-gray-600">Operation</span>
            <div class="inline-flex border rounded overflow-hidden">
              <button type="button" class={`px-2 py-1 ${ops[i-1]==='AND'?'bg-blue-600 text-white':'bg-white'}`} on:click={() => ops[i-1] = 'AND'}>AND</button>
              <button type="button" class={`px-2 py-1 ${ops[i-1]==='OR'?'bg-blue-600 text-white':'bg-white'}`} on:click={() => ops[i-1] = 'OR'}>OR</button>
            </div>
          </div>
        {/if}
        <div class="border rounded bg-white">
          <ConceptTolerance name={(c.name || c.method) + ''}
                            items={labeledItemsForConcept(c)}
                            width={520}
                            height={300}
                            p={(tolById.get(c.id)?.p ?? 50)}
                            active={(tolById.get(c.id)?.active ?? true)}
                            currentPos={(i>0 && ops[i-1]==='OR') ? (inferredById.get(concepts[i-1].id)?.posIds || []) : []}
                            currentNeg={(i>0 && ops[i-1]==='AND') ? (inferredById.get(concepts[i-1].id)?.negIds || []) : []}
                            on:inferred={(e) => onInferred(c.id, e.detail)} />
        </div>
      {/each}
    </div>
  </div>
  {/key}
{/if}

<!-- Save combined concept at the bottom -->
<div class="mt-4 flex items-center gap-2">
  <input class="px-2 py-1 border rounded text-sm flex-1 bg-white" placeholder="Name combined concept" bind:value={combinedName} />
  <button class="btn" on:click={() => {
      const ids = combinedGood
      if (!ids || ids.length === 0) return
      const chain = { conceptIds: concepts.map(c => c.id), ops: [...ops], tolerances: concepts.map(c => ({ id: c.id, p: Number(tolById.get(c.id)?.p ?? 50), active: Boolean(tolById.get(c.id)?.active ?? true) })) }
      dispatch('create', { name: (combinedName||'Combined').trim(), method: defaultMethod, good: ids, bad: [], chain })
      combinedName = ''
    }} disabled={(combinedGood||[]).length===0}>
    <span class="i-heroicons-plus-circle mr-1" /> Save Combined
  </button>
</div>

<style>
</style>
