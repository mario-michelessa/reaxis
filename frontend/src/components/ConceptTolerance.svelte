<script>
  import { slide } from 'svelte/transition'
  import { createEventDispatcher } from 'svelte'
  // ConceptTolerance: visualize labeled/unlabeled examples in 2D with a tolerance slider.
  // Props
  export let name = 'Concept'
  // items: [{ id, url, x, y, label?: 'pos'|'neg' }]
  export let items = []
  export let width = 700
  export let height = 420
  export let padding = 20
  // Slider parameter p controls classification tendency
  // Dual mode:
  // - If both pos and neg exist: p is 0..100 UI percent mapped to ratio r in [0.1, 10] via r = 10^((p-50)/50)
  // - If only one class exists: p is 0..100 UI percent mapped to distance threshold t in [0, maxD], where 0 selects only labeled, 100 selects everyone
  export let p = 50 // UI percent 0..100
  // UI state
  let collapsed = false
  export let active = true
  export let currentPos = [] // array of ids to force as pos (dark green)
  export let currentNeg = [] // array of ids to force as neg (dark red)
  const dispatch = createEventDispatcher()

  // Derived sets
  $: posItems = items.filter(i => i.label === 'pos')
  $: negItems = items.filter(i => i.label === 'neg')
  $: currentPosSet = new Set(currentPos || [])
  $: currentNegSet = new Set(currentNeg || [])
  $: unlabeledItems = items.filter(i => i.label !== 'pos' && i.label !== 'neg' && !currentPosSet.has(i.id) && !currentNegSet.has(i.id))

  function dist(a, b) {
    const dx = Number(a.x ?? 0) - Number(b.x ?? 0)
    const dy = Number(a.y ?? 0) - Number(b.y ?? 0)
    return Math.hypot(dx, dy)
  }

  function minDistToSet(pt, setArr) {
    if (!setArr || setArr.length === 0) return Infinity
    let best = Infinity
    for (const s of setArr) {
      const d = dist(pt, s)
      if (d < best) best = d
    }
    return best
  }

  // Compute nearest distances for all points
  $: dists = items.map((it) => {
    const d_pos = minDistToSet(it, posItems)
    const d_neg = minDistToSet(it, negItems)
    return { id: it.id, d_pos, d_neg }
  })

  function byId(id) { return items.find(i => i.id === id) }

  // Helper: map UI percent [0..100] to ratio in [0.1..10]
  function percentToRatio(percent) {
    const clamped = Math.max(0, Math.min(100, Number(percent || 0)))
    return Math.pow(10, (clamped - 50) / 50)
  }

  // Classification for unlabeled based on slider p
  $: inferred = active
    ? unlabeledItems.map((it) => {
        const ds = dists.find(d => d.id === it.id)
        const dpos = ds?.d_pos ?? Infinity
        const dneg = ds?.d_neg ?? Infinity
        let inferredLabel = null
        const hasPos = posItems.length > 0
        const hasNeg = negItems.length > 0
        if (!hasPos && !hasNeg) {
          inferredLabel = null
        } else if (hasPos && hasNeg) {
          const r = percentToRatio(p)
          inferredLabel = (dpos < r * dneg) ? 'pos' : 'neg'
        } else if (hasPos && !hasNeg) {
          // Only positives labeled: use distance to positives as tolerance.
          // p=0 selects only labeled (threshold 0), p=100 selects everyone (threshold = max d_pos).
          const maxDpos = dists.reduce((m, d) => (d.d_pos > m ? d.d_pos : m), 0)
          const threshold = (Math.max(0, Math.min(100, Number(p || 0))) / 100) * maxDpos
          inferredLabel = (dpos <= threshold) ? 'pos' : 'neg'
        } else if (hasNeg && !hasPos) {
          // Only negatives labeled: use distance to negatives as tolerance (converse).
          const maxDneg = dists.reduce((m, d) => (d.d_neg > m ? d.d_neg : m), 0)
          const threshold = (Math.max(0, Math.min(100, Number(p || 0))) / 100) * maxDneg
          inferredLabel = (dneg <= threshold) ? 'neg' : 'pos'
        }
        return { ...it, inferredLabel, d_pos: dpos, d_neg: dneg }
      })
    : []

  $: inferredPos = inferred.filter(i => i.inferredLabel === 'pos')
  $: inferredNeg = inferred.filter(i => i.inferredLabel === 'neg')

  // Emit inferred sets so parent can chain boolean logic, but only when changed
  function arraysEqual(a = [], b = []) {
    if (a === b) return true
    if (!Array.isArray(a) || !Array.isArray(b)) return false
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
    return true
  }
  let lastEmittedPos = []
  let lastEmittedNeg = []
  let lastEmittedP = p
  let lastEmittedActive = active
  $: (function emitIfChanged() {
    const posIds = inferredPos.map(i => i.id)
    const negIds = inferredNeg.map(i => i.id)
    const changed = !arraysEqual(posIds, lastEmittedPos) || !arraysEqual(negIds, lastEmittedNeg) || lastEmittedP !== p || lastEmittedActive !== active
    if (changed) {
      try { console.log('[ConceptTolerance] emit inferred', { name, pos: posIds.length, neg: negIds.length, p, active, currentPos: (currentPos||[]).length, currentNeg: (currentNeg||[]).length }) } catch (_) {}
      lastEmittedPos = posIds
      lastEmittedNeg = negIds
      lastEmittedP = p
      lastEmittedActive = active
      dispatch('inferred', { posIds, negIds, p, active })
    }
  })()

  // Example selections for right-side panel
  // Labeled examples (up to 2)
  $: labeledPosShow = posItems.slice(0, 2)
  $: labeledNegShow = negItems.slice(0, 2)
  // Farthest inferred from their own class centers
  $: farPosShow = [...inferredPos]
    .sort((a, b) => (b.d_pos - a.d_pos))
    .slice(0, 2)
  $: farNegShow = [...inferredNeg]
    .sort((a, b) => (b.d_neg - a.d_neg))
    .slice(0, 2)

  // Plot helpers
  $: innerW = Math.max(0, width - 2 * padding)
  $: innerH = Math.max(0, height - 2 * padding)
  function pxX(x) { return padding + Math.max(0, Math.min(1, Number(x ?? 0))) * innerW }
  function pxY(y) { return padding + Math.max(0, Math.min(1, Number(y ?? 0))) * innerH }

  // Colors
  const cPos = '#22c55e' // green-500
  const cNeg = '#ef4444' // red-500
  const cPosDark = '#15803d' // green-700
  const cNegDark = '#b91c1c' // red-700
  const cPosLight = '#bbf7d0' // green-200
  const cNegLight = '#fecaca' // red-200
  const cUnk = '#94a3b8' // slate-400

  // Include dependencies as parameters so Svelte tracks updates
  function colorFor(it, pDep, activeDep, inferredLenDep, posSetSizeDep, negSetSizeDep) {
    if (!activeDep) return cUnk
    if (currentPosSet.has(it.id)) return cPosDark
    if (currentNegSet.has(it.id)) return cNegDark
    if (it.label === 'pos') return cPos
    if (it.label === 'neg') return cNeg
    const inf = inferred.find(j => j.id === it.id)
    if (inf?.inferredLabel === 'pos') return cPosLight
    if (inf?.inferredLabel === 'neg') return cNegLight
    return cUnk
  }

  // Slider UI range: 0..100 percent
  const sliderMin = 0
  const sliderMax = 100
  const sliderStep = 1
</script>

<div class="w-full border rounded bg-white overflow-hidden">
  <!-- Ribbon header -->
  <div class="bg-gradient-to-r from-indigo-500 to-purple-500 text-white px-2 py-1.5 text-sm font-medium flex items-center justify-between">
    <button type="button" class="px-1 py-0.5 rounded hover:bg-white/10" on:click={() => (collapsed = !collapsed)} aria-label={collapsed ? 'Expand' : 'Collapse'}>
      {#if collapsed}▶{:else}▼{/if}
    </button>
    <div class="px-1 truncate">{name}</div>
    <button type="button" class="px-2 py-0.5 text-xs rounded bg-white/20 hover:bg-white/30" on:click={() => (active = !active)} aria-label={active ? 'Deactivate' : 'Activate'}>
      {#if active}Deactivate{:else}Activate{/if}
    </button>
  </div>

  {#if !collapsed}
  <div class="flex gap-3 p-3" transition:slide>
    <!-- Left: Scatterplot and slider -->
    <div class="flex-1 min-w-0">
      <svg {width} {height} role="img" aria-label="Concept scatterplot" class="block border border-gray-200 rounded bg-white">
        <!-- Axes box -->
        <rect x={padding} y={padding} width={innerW} height={innerH} fill="white" stroke="#e5e7eb" />
        <!-- Points -->
        {#each items as it (it.id)}
          <circle cx={pxX(it.x)} cy={pxY(it.y)} r="4" fill={colorFor(it, p, active, inferred.length, currentPosSet.size, currentNegSet.size)} />
        {/each}
      </svg>

      <!-- Slider -->
      <div class="mt-3">
        <label class="text-xs text-gray-700 flex items-center gap-2">
          <span>Tolerance</span>
          <input type="range" min={sliderMin} max={sliderMax} step={sliderStep} bind:value={p} class="flex-1" disabled={!active} />
          <span class="tabular-nums">{Math.round(p)}%</span>
        </label>
        {#if posItems.length > 0 && negItems.length > 0}
          <div class="mt-1 text-[11px] text-gray-500">
            Rule: classify pos when d_pos &lt; r · d_neg; r = 10^((p-50)/50)
          </div>
        {:else if posItems.length > 0}
          <div class="mt-1 text-[11px] text-gray-500">
            Only positives labeled — threshold on d_pos: p=0 selects only labeled, p=100 selects everyone
          </div>
        {:else if negItems.length > 0}
          <div class="mt-1 text-[11px] text-gray-500">
            Only negatives labeled — threshold on d_neg: p=0 selects only labeled, p=100 selects everyone
          </div>
        {/if}
        {#if !active}
          <div class="mt-1 text-[11px] text-gray-500">Concept deactivated — all points shown as unclassified.</div>
        {/if}
        <div class="mt-1 text-[11px] text-gray-400">
          Colors — pos: {cPos}, neg: {cNeg}, inferred pos: {cPosLight}, inferred neg: {cNegLight}
        </div>
      </div>
    </div>

    <!-- Right: Example thumbnails -->
    <div class="w-64 shrink-0">
      <div class="text-xs text-gray-700 mb-1">Positives</div>
      <div class="grid grid-cols-2 gap-2 mb-3">
        {#each labeledPosShow as it (it.id)}
          <img alt="" src={it.url} class="w-full h-20 object-cover rounded border" loading="lazy" />
        {/each}
        {#if labeledPosShow.length === 0}
          <div class="col-span-2 text-[11px] text-gray-400">No labeled positives</div>
        {/if}
      </div>
      <div class="text-xs text-gray-700 mb-1">Farthest inferred positives</div>
      <div class="grid grid-cols-2 gap-2 mb-4">
        {#each farPosShow as it (it.id)}
          <img alt="" src={it.url} class="w-full h-20 object-cover rounded border" loading="lazy" />
        {/each}
        {#if farPosShow.length === 0}
          <div class="col-span-2 text-[11px] text-gray-400">No inferred positives</div>
        {/if}
      </div>

      <div class="text-xs text-gray-700 mb-1">Negatives</div>
      <div class="grid grid-cols-2 gap-2 mb-3">
        {#each labeledNegShow as it (it.id)}
          <img alt="" src={it.url} class="w-full h-20 object-cover rounded border" loading="lazy" />
        {/each}
        {#if labeledNegShow.length === 0}
          <div class="col-span-2 text-[11px] text-gray-400">No labeled negatives</div>
        {/if}
      </div>
      <div class="text-xs text-gray-700 mb-1">Farthest inferred negatives</div>
      <div class="grid grid-cols-2 gap-2">
        {#each farNegShow as it (it.id)}
          <img alt="" src={it.url} class="w-full h-20 object-cover rounded border" loading="lazy" />
        {/each}
        {#if farNegShow.length === 0}
          <div class="col-span-2 text-[11px] text-gray-400">No inferred negatives</div>
        {/if}
      </div>
    </div>
  </div>
  {/if}
</div>

<style>
</style>
