<script>
  import { createEventDispatcher } from 'svelte'
  import { slide } from 'svelte/transition'
  import DiftPartSelector from './DiftPartSelector.svelte'
  import ColorSpaceSelector from './ColorSpaceSelector.svelte'
  // AxesPanel: list and manage 1D axes (scores per image id in [0,1])
  // props:
  // - axes: [{ id, name, coords: Record<string, number> }]
  // - items: optional [{ id, url }] for eventual thumbnails (unused for now)
  // - concepts: optional [{ id, name, method, pos: string[], neg: string[] }]
  // - embedSelection: current projection selection; emits 'embedChange' when changed
  // - diftPart: current dift local composition part (e.g., '11')
  export let axes = []
  // export let items = []
  export let embedSelection = 'color_rgb'
  // export let labels = new Map()
  // Projections forwarded from App: [{ value, label }]
  export let projections = []
  // Custom projections from App: [{ id, name, xAxisId, yAxisId }]
  export let customProjections = []
  const dispatch = createEventDispatcher()

  function onDelete(ax) { dispatch('delete', { id: ax.id }) }
  function onRename(ax) {
    const next = prompt('Rename axis', ax.name || 'Axis')
    if (next && next.trim() && next !== ax.name) dispatch('rename', { id: ax.id, name: next.trim() })
  }

  // Drag support: set transferable axis id
  function onDragStart(e, ax) {
    try {
      e.dataTransfer.setData('application/axis-id', ax.id)
      e.dataTransfer.setData('text/plain', ax.id)
      e.dataTransfer.effectAllowed = 'copyMove'
    } catch (_) {}
  }

  // Availability of metadata axes
  $: hasMetaAxes = (axes || []).some(a => (a && (a.group === 'meta' || (a.id && a.id.startsWith('axis:meta:')))))

  // Projection selection (vertical radios) provided by parent
  function onProjectionChange(val) {
    console.log('[AxesPanel] projection change', val)
    if (!val) return
    dispatch('embedChange', { selection: val })
  }

  // Collapse state per projection/custom id
  let collapsed = {}
  function toggleCollapsed(key) {
    try { console.log('[AxesPanel] toggle', key, !!collapsed[key]) } catch (_) {}
    collapsed = { ...collapsed, [key]: !collapsed[key] }
  }

  // Map a projection value to its grouping key (projection-level)
  function methodForProjectionValue(val) {
    // Built-ins group by projection, not variant
    if (val === 'clip') return 'clip'
    if (val === 'color_rgb') return 'color_rgb'
    if (val === 'shape') return 'shape'
    
    if (val === 'meta') return 'meta'
    // Customs: id is the group key
    return val
  }

  function idMatchesProjection(groupKey, axisId) {
    if (!axisId || !groupKey) return false
    if (groupKey === 'color_rgb') return axisId.startsWith('axis:color_lch:') || axisId.startsWith('axis:color_hsv:') || axisId.startsWith('axis:color_rgb:')
    if (groupKey === 'shape') return axisId.startsWith('axis:dino:') || axisId.startsWith('axis:dift_sd_part') || axisId.startsWith('axis:dift_sd:')
    if (groupKey === 'clip') return axisId.startsWith('axis:clip:')
    
    if (groupKey === 'meta') return axisId.startsWith('axis:meta:')
    // For customs, prefer explicit group match; id prefix not applicable
    return false
  }

  // $: posCount = (() => {
  //   let c = 0
  //   if (!labels) return 0
  //   try {
  //     if (labels instanceof Map) { labels.forEach((v) => { if (v === 'pos') c++ }) }
  //     else { for (const v of Object.values(labels)) if (v === 'pos') c++ }
  //   } catch (_) {}
  //   return c
  // })()
  // $: negCount = (() => {
  //   let c = 0
  //   if (!labels) return 0
  //   try {
  //     if (labels instanceof Map) { labels.forEach((v) => { if (v === 'neg') c++ }) }
  //     else { for (const v of Object.values(labels)) if (v === 'neg') c++ }
  //   } catch (_) {}
  //   return c
  // })()

  // function createAxisFromLabels(targetGroup = null) {
  //   // Gather positives/negatives based on current labels
  //   const posIds = new Set()
  //   const negIds = new Set()
  //   if (labels) {
  //     if (labels instanceof Map) {
  //       labels.forEach((v, k) => { if (v === 'pos') posIds.add(k); if (v === 'neg') negIds.add(k) })
  //     } else {
  //       for (const [k, v] of Object.entries(labels)) { if (v === 'pos') posIds.add(k); if (v === 'neg') negIds.add(k) }
  //     }
  //   }
  //   const posItems = (items || []).filter(i => posIds.has(i.id))
  //   const negItems = (items || []).filter(i => negIds.has(i.id))
  //   if (posItems.length === 0 && negItems.length === 0) {
  //     alert('Select positive and negative examples using the lasso tool')
  //     return
  //   }
  //   function dist(a, b) { return Math.hypot((Number(a.x||0) - Number(b.x||0)), (Number(a.y||0) - Number(b.y||0))) }
  //   function minDist(pt, arr) { if (!arr.length) return Infinity; let m = Infinity; for (const s of arr) { const d = dist(pt, s); if (d < m) m = d } return m }
  //   const coords = {}
  //   let maxDp = 0, maxDn = 0
  //   for (const it of (items || [])) { const dp = minDist(it, posItems); const dn = minDist(it, negItems); if (isFinite(dp) && dp > maxDp) maxDp = dp; if (isFinite(dn) && dn > maxDn) maxDn = dn }
  //   function clamp01(v) { return Math.max(0, Math.min(1, v)) }
  //   for (const it of (items || [])) {
  //     const dp = minDist(it, posItems)
  //     const dn = minDist(it, negItems)
  //     let s = 0.5
  //     if (posItems.length && negItems.length) s = clamp01(dn / (dn + dp + 1e-9))
  //     else if (posItems.length && !negItems.length) s = clamp01(1 - (dp / (maxDp + 1e-9)))
  //     else if (negItems.length && !posItems.length) s = clamp01(dn / (maxDn + 1e-9))
  //     coords[it.id] = s
  //   }
  //   const proposed = prompt('Name this axis', 'Axis')
  //   if (!proposed || !proposed.trim()) return
  //   const id = `axis:labels:${Date.now()}`
  //   // Pass through an optional grouping key so parent can associate axis with a projection
  //   const group = targetGroup || methodForProjectionValue(embedSelection)
  //   dispatch('create', { id, name: proposed.trim(), coords, group })
  // }

</script>

<!-- Projections list with axes grouped under each projection -->
<div class="tile">
  <div class="tile-content">
    <div class="text-sm text-gray-700 mb-1">Projections</div>
    <div class="grid gap-2">
      {#each projections as opt}
        {#key opt.value}
        <div class="subtile">
          <div class="flex items-center gap-2 {opt.value==='meta' && !hasMetaAxes ? 'opacity-50' : ''}"
               role="button" tabindex="0"
               on:click={(e)=>{ const disabled = (opt.value==='meta' && !hasMetaAxes); if (disabled) return; if (!(e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON'))) { collapsed = { ...collapsed, [opt.value]: !collapsed[opt.value] } } }}
               on:keydown={(e)=>{ const disabled = (opt.value==='meta' && !hasMetaAxes); if (disabled) return; if (e.key==='Enter' || e.key===' ') { e.preventDefault(); collapsed = { ...collapsed, [opt.value]: !collapsed[opt.value] } } }}>
            <input type="radio" name="axes-proj" value={opt.value} checked={embedSelection===opt.value} disabled={opt.value==='meta' && !hasMetaAxes} on:change={(e) => onProjectionChange(e.currentTarget?.value)} />
            <button type="button" class="btn btn-xs btn-ui-secondary" disabled={opt.value==='meta' && !hasMetaAxes} on:click|stopPropagation|preventDefault={() => { const disabled = (opt.value==='meta' && !hasMetaAxes); if (disabled) return; collapsed = { ...collapsed, [opt.value]: !collapsed[opt.value] } }} aria-label="Collapse" aria-expanded={!collapsed[opt.value]} aria-controls={`proj-body-${opt.value}`}>
              {collapsed[opt.value] ? '▶' : '▼'}
            </button>
            <span class="text-base">{opt.label}</span>
          </div>
          {#if opt.value === 'color_rgb' && !collapsed[opt.value]}
            <div class="subtile mt-2">
              <div class="text-sm text-gray-700 mb-1">Color spaces</div>
              <ColorSpaceSelector on:change={(e) => onProjectionChange(e.detail)} />
            </div>
          {/if}
          <div id={`proj-body-${opt.value}`}
               class="collapse-body"
               style={`overflow:hidden; transition:max-height 0.2s ease, opacity 0.2s ease; max-height:${(opt.value==='meta' && !hasMetaAxes) || collapsed[opt.value]?'0px':'1600px'}; opacity:${(opt.value==='meta' && !hasMetaAxes) || collapsed[opt.value]?'0':'1'};`}>
              {#if opt.value === 'shape'}
                <div class="subtile mt-2">
                  <div class="text-sm text-gray-700 mb-1">Which part?</div>
                  <DiftPartSelector on:change={(e) => onProjectionChange(e.detail)} />
                </div>
              {/if}
              <!-- Axes under this projection -->
              {#if axes && axes.length > 0}
                    <div class="mt-2 grid gap-2">
                      {#each axes.filter(a => (a?.group === methodForProjectionValue(opt.value)) || (a?.id && idMatchesProjection(methodForProjectionValue(opt.value), a.id))) as ax (ax.id)}
                        <div class="subtile flex items-center justify-between gap-2" draggable={true} on:dragstart={(e)=>onDragStart(e, ax)} title="Drag onto X/Y drop zones">
                          <div class="min-w-0 flex-1">
                            <div class="text-base font-medium truncate">{ax.name.slice(0, 218) || 'Axis'}</div>
                          </div>
                          <div class="shrink-0 inline-flex gap-1 items-center">
                            <div class="text-sm text-gray-500">Set: </div>
                            <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => dispatch('setX', { id: ax.id })} title="Use as X axis">X</button>
                            <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => dispatch('setY', { id: ax.id })} title="Use as Y axis">Y</button>
                            <div class="text-sm text-gray-500"> | </div>
                            <button type="button" class="btn btn-xs btn-ui-secondary" on:click={() => onRename(ax)} title="Rename">✎</button>
                            <button type="button" class="btn btn-xs btn-danger" on:click={() => onDelete(ax)} title="Delete">✕</button>
                          </div>
                        </div>
                      {/each}
                      
                    </div>
              {/if}
          </div>
        </div>
        {/key}
      {/each}

      <!-- Save current projection button -->
      <button type="button"
              class="subtile flex items-center justify-between gap-2 text-left mt-2"
              on:click={() => {
                const name = prompt('Name this projection', 'Projection')
                if (!name || !name.trim()) return
                dispatch('addCustomProjection', { name: name.trim() })
              }}
              title="Save current projection">
        <div class="min-w-0 flex-1">
          <div class="text-base font-medium truncate flex items-center gap-2">
            <span class="i-heroicons-plus-circle text-slate-600" /> Save current projection
          </div>
          <div class="text-sm text-gray-500 truncate">Saves the currently assigned X and Y axes.</div>
        </div>
        <div class="shrink-0 inline-flex gap-1 items-center">
          <span class="btn btn-xs btn-primary">Save</span>
        </div>
      </button>

      <!-- List custom projections -->
      {#each customProjections as cp}
        {#key cp.id}
        <div class="subtile mt-2">
          <div class="flex items-center gap-2" role="button" tabindex="0" on:click={(e)=>{ if (!(e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON'))) { collapsed = { ...collapsed, [cp.id]: !collapsed[cp.id] } } }} on:keydown={(e)=>{ if (e.key==='Enter' || e.key===' ') { e.preventDefault(); collapsed = { ...collapsed, [cp.id]: !collapsed[cp.id] } } }}>
            <input type="radio" name="axes-proj" value={cp.id} checked={embedSelection===cp.id} on:change={(e) => onProjectionChange(e.currentTarget?.value)} />
            <button type="button" class="btn btn-xs btn-ui-secondary" on:click|stopPropagation|preventDefault={() => { collapsed = { ...collapsed, [cp.id]: !collapsed[cp.id] } }} aria-label="Collapse" aria-expanded={!collapsed[cp.id]} aria-controls={`proj-body-${cp.id}`}>
              {collapsed[cp.id] ? '▶' : '▼'}
            </button>
            <span class="text-base">{cp.name}</span>
          </div>
          <div id={`proj-body-${cp.id}`}
               class="collapse-body"
               style={`overflow:hidden; transition:max-height 0.2s ease, opacity 0.2s ease; max-height:${collapsed[cp.id]?'0px':'1600px'}; opacity:${collapsed[cp.id]?'0':'1'};`}>
            <div class="mt-2 grid gap-1 text-sm text-gray-700">
              <div>Assigned X axis: <span class="font-medium">{(axes||[]).find(a => a.id===cp.xAxisId)?.name || cp.xAxisId}</span></div>
              <div>Assigned Y axis: <span class="font-medium">{(axes||[]).find(a => a.id===cp.yAxisId)?.name || cp.yAxisId}</span></div>
              <div class="text-xs text-gray-500">Selecting this projection assigns these axes.</div>
            </div>
          </div>
        </div>
        {/key}
      {/each}
    </div>
  </div>
</div>

<style>
  .collapse-body {}
</style>
