<script>
  import { createEventDispatcher } from 'svelte'
  // AxisBuilder: pick positive/negative examples on a scatter, tune tolerance, then create a 1D axis
  export let items = [] // [{id,url,x,y}]
  export let width = 520
  export let height = 320
  export let padding = 20
  export let p = 50 // tolerance percent (0..100)

  const dispatch = createEventDispatcher()
  let name = 'New Axis'

  // Label map: id -> 'pos'|'neg'
  let labels = new Map()
  function toggleLabel(id, mode = 'cycle') {
    const prev = labels.get(id)
    let next
    if (mode === 'pos') next = (prev === 'pos') ? undefined : 'pos'
    else if (mode === 'neg') next = (prev === 'neg') ? undefined : 'neg'
    else {
      next = prev === undefined ? 'pos' : (prev === 'pos' ? 'neg' : undefined)
    }
    const m = new Map(labels)
    if (next) m.set(id, next); else m.delete(id)
    labels = m
  }

  $: posItems = items.filter(i => labels.get(i.id) === 'pos')
  $: negItems = items.filter(i => labels.get(i.id) === 'neg')

  function dist(a, b) { return Math.hypot((Number(a.x||0)-Number(b.x||0)), (Number(a.y||0)-Number(b.y||0))) }
  function minDistToSet(pt, setArr) { if (!setArr || setArr.length === 0) return Infinity; let best = Infinity; for (const s of setArr) { const d = dist(pt, s); if (d < best) best = d } return best }
  $: dists = items.map(it => ({ id: it.id, d_pos: minDistToSet(it, posItems), d_neg: minDistToSet(it, negItems) }))

  function percentToRatio(percent) { const clamped = Math.max(0, Math.min(100, Number(percent || 0))); return Math.pow(10, (clamped - 50) / 50) }
  function clamp01(v) { return Math.max(0, Math.min(1, v)) }

  // Continuous score in [0,1]: ~0 near negatives, ~1 near positives
  function scoreFor(it) {
    const ds = dists.find(d => d.id === it.id) || { d_pos: Infinity, d_neg: Infinity }
    const dp = ds.d_pos, dn = ds.d_neg
    const hasPos = posItems.length > 0
    const hasNeg = negItems.length > 0
    if (!hasPos && !hasNeg) return 0.5
    if (hasPos && hasNeg) {
      const r = percentToRatio(p)
      return clamp01(dn / (dn + r * dp + 1e-9))
    } else if (hasPos && !hasNeg) {
      const maxDp = dists.reduce((m, d) => (d.d_pos > m ? d.d_pos : m), 0) || 1
      return clamp01(1 - (dp / (maxDp + 1e-9)))
    } else { // hasNeg only
      const maxDn = dists.reduce((m, d) => (d.d_neg > m ? d.d_neg : m), 0) || 1
      return clamp01(dn / (maxDn + 1e-9))
    }
  }

  $: innerW = Math.max(0, width - 2 * padding)
  $: innerH = Math.max(0, height - 2 * padding)
  function pxX(x) { return padding + Math.max(0, Math.min(1, Number(x ?? 0))) * innerW }
  function pxY(y) { return padding + Math.max(0, Math.min(1, Number(y ?? 0))) * innerH }

  // Colors
  const cPos = '#16a34a' // green-600
  const cNeg = '#dc2626' // red-600
  const cUnk = '#64748b' // slate-500
  function colorFor(it) {
    const l = labels.get(it.id)
    if (l === 'pos') return cPos
    if (l === 'neg') return cNeg
    // gradient based on score
    const s = scoreFor(it)
    // mix red->gray->green roughly
    const r = Math.round((1 - s) * 220 + s * 100)
    const g = Math.round((1 - s) * 38 + s * 163)
    const b = Math.round((1 - s) * 38 + s * 74)
    return `rgb(${r},${g},${b})`
  }

  function onPointClick(it, e) {
    if (e && (e.metaKey || e.ctrlKey)) toggleLabel(it.id, 'neg')
    else toggleLabel(it.id, 'pos')
  }

  function createAxis() {
    if (posItems.length === 0 && negItems.length === 0) return
    const coords = {}
    for (const it of items) coords[it.id] = Number(scoreFor(it))
    const id = `axis:${Date.now()}`
    dispatch('create', { id, name: (name||'Axis').trim(), coords })
    // reset labels but keep name
    labels = new Map()
  }
</script>

<div class="w-full border rounded bg-white overflow-hidden">
  <div class="bg-gradient-to-r from-indigo-500 to-purple-500 text-white px-2 py-1.5 text-sm font-medium flex items-center justify-between">
    <div class="px-1 truncate">Axis Builder</div>
    <div class="text-[11px] opacity-90">Click: +pos • Ctrl/Cmd+Click: +neg</div>
  </div>

  <div class="p-3">
    <svg {width} {height} role="img" aria-label="Axis builder scatter" class="block border border-gray-200 rounded bg-white">
      <rect x={padding} y={padding} width={innerW} height={innerH} fill="white" stroke="#e5e7eb" />
      {#each items as it (it.id)}
        <circle cx={pxX(it.x)} cy={pxY(it.y)} r="4" fill={colorFor(it)} on:click={(e) => onPointClick(it, e)} />
      {/each}
    </svg>

    <div class="mt-3 flex items-center gap-3">
      <label class="text-xs text-gray-700 flex items-center gap-2">
        <span>Tolerance</span>
        <input type="range" min="0" max="100" step="1" bind:value={p} class="flex-1" />
        <span class="tabular-nums">{Math.round(p)}%</span>
      </label>
      <div class="text-xs text-gray-500">Positives: {posItems.length} • Negatives: {negItems.length}</div>
    </div>

    <div class="mt-3 flex items-center gap-2">
      <input class="px-2 py-1 border rounded text-sm flex-1 bg-white" placeholder="Axis name" bind:value={name} />
      <button class="px-2 py-1 text-sm rounded bg-blue-600 text-white disabled:opacity-50" on:click={createAxis} disabled={posItems.length===0 && negItems.length===0}>Create Axis</button>
    </div>
  </div>
</div>

<style>
</style>

