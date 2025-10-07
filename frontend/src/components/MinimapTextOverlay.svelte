<script>
  import { createEventDispatcher } from 'svelte'

  // Action to bind elements to rectInputs array
  function bindRectInput(node, index) {
    rectInputs[index] = node;
    return {
      update(newIndex) {
        rectInputs[newIndex] = node;
      },
      destroy() {
        rectInputs[index] = null;
      }
    };
  }

  export let width = 220
  export let height = 220
  export let color = '#10b981'
  export let rectangles = [] // [{ x,y,w,h, text, color? }], normalized 0..1
  export let defaultText = ''
  export let handleSize = 30 // px
  export let handleHitFactor = 3 // hit tolerance multiplier

  const dispatch = createEventDispatcher()

  let host
  let placing = false
  let dragging = false
  let activeIdx = -1
  let draft = null // { x,y,w,h, text }
  let dragOff = { x: 0, y: 0 }
  let draftInput
  let rectInputs = []

  function posFromEvent(e) {
    const r = host.getBoundingClientRect()
    const w = r.width || width
    const h = r.height || height
    const x = Math.max(0, Math.min(1, (e.clientX - r.left) / w))
    const y = Math.max(0, Math.min(1, (e.clientY - r.top) / h))
    return { x, y }
  }

  export function startPlacing() {
    placing = true
    draft = null
    activeIdx = -1
  }

  export function clearRectangles() {
    rectangles = []
  }

  function mousedown(e) {
    // Ignore UI controls (inputs/buttons) to allow text editing without dragging
    const target = e.target
    if (target && (target.closest('.text-ui') || target.tagName === 'INPUT' || target.tagName === 'BUTTON' || target.tagName === 'TEXTAREA')) {
      return
    }
    const pos = posFromEvent(e)
    // Priority: existing handle drag over placing
    const idx = rectangles.findIndex(r => pos.x >= r.x && pos.x <= r.x + r.w && pos.y >= r.y && pos.y <= r.y + r.h)
    if (idx >= 0 && isOnHandle(pos, rectangles[idx])) {
      activeIdx = idx
      dragging = true
      dragOff = { x: pos.x - rectangles[activeIdx].x, y: pos.y - rectangles[activeIdx].y }
      placing = false
      return
    }
    if (placing) {
      draft = { x: pos.x, y: pos.y, w: 0.1, h: 0.04, text: defaultText }
      return
    }
    // If not placing and not on handle, do nothing
  }

  function mousemove(e) {
    if (!dragging) return
    const pos = posFromEvent(e)
    if (activeIdx >= 0) {
      const r = rectangles[activeIdx]
      let nx = pos.x - dragOff.x
      let ny = pos.y - dragOff.y
      nx = Math.max(0, Math.min(1 - r.w, nx))
      ny = Math.max(0, Math.min(1 - r.h, ny))
      rectangles = rectangles.map((rr, i) => i === activeIdx ? { ...rr, x: nx, y: ny } : rr)
    }
  }

  function mouseup() {
    dragging = false
  }

  function confirmDraft() {
    if (!draft) return
    rectangles = [...rectangles, { ...draft }]
    dispatch('confirmRegion', { rect: { x: draft.x, y: draft.y, w: draft.w, h: draft.h }, text: draft.text })
    draft = null
    placing = false
  }

  function cancelDraft() {
    draft = null
    placing = false
  }
  
  function isOnHandle(pos, r) {
    const rect = host ? host.getBoundingClientRect() : { width, height }
    const tol = (handleSize * handleHitFactor) / Math.max(rect.width || width, rect.height || height)
    const hx = r.x
    const hy = r.y
    return (Math.abs(pos.x - hx) <= tol && Math.abs(pos.y - hy) <= tol)
  }

  function updateDraftSize() {
    if (!draft || !draftInput) return
    const pw = draftInput.offsetWidth
    const ph = draftInput.offsetHeight
    const minW = 100 // px
    draft.w = Math.min(1, Math.max(minW / width, pw / width))
    draft.h = Math.min(1, Math.max(0.04, ph / height))
  }

  function updateRectSize(e, i) {
    const el = e.currentTarget
    if (!el) return
    const pw = el.offsetWidth
    const ph = el.offsetHeight
    const txt = el.textContent || ''
    const minW = 100 // px
    rectangles = rectangles.map((rr, idx) => idx === i ? { ...rr, text: txt, w: Math.min(1, Math.max(minW / width, pw / width)), h: Math.min(1, Math.max(0.04, ph / height)) } : rr)
  }
</script>

<div bind:this={host}
     class="absolute inset-0 z-20 pointer-events-auto select-none"
     on:mouseup|preventDefault={mouseup}
     on:mouseleave|preventDefault={mouseup}
     on:mousedown|preventDefault={mousedown}
     on:mousemove|preventDefault={mousemove}
>
  <!-- Existing rectangles -->
  {#each rectangles as r, i}
    <div class="absolute" style={`left:${r.x*100}%;top:${r.y*100}%;width:${r.w*100}%;height:${r.h*100}%` }>
      <div class="absolute inset-0 rounded-full" style={`border:${2}px solid ${r.color||color};background:${(r.color||color)}33;box-shadow:0 4px 12px 0 rgba(0,0,0,0.08);`} />
      <!-- drag handle at top-left -->
      <div class="absolute bg-white border rounded-full cursor-move" style={`top:0;left:0;transform:translate(-0%,-0%);width:${handleSize}px;height:${handleSize}px;border:1px solid #94a3b8;z-index:30;`} />
      <div class="text-ui absolute inset-0 flex items-center justify-center" on:mousedown|stopPropagation on:click|stopPropagation>
        <div contenteditable="true" class="mx-[5px] my-[5px] text-[18px]  px-2 py-1 rounded border  shadow-sm overflow-hidden whitespace-nowrap"
             on:input={(e) => updateRectSize(e, i)}>{r.text}</div>
      </div>
    </div>
  {/each}

  <!-- Draft rectangle and toolbar -->
  {#if draft}
    <div class="absolute" style={`left:${draft.x*100}%;top:${draft.y*100}%;width:${draft.w*100}%;height:${draft.h*100}%` }>
      <div class="absolute inset-0 rounded-full" style={`border:${2}px solid ${color};background:${color}33;box-shadow:0 4px 12px 0 rgba(0,0,0,0.08);`} />
      <div class="text-ui absolute inset-0 flex items-center justify-center" on:mousedown|stopPropagation on:click|stopPropagation>
        <input bind:this={draftInput}
               class="mx-[10px] my-[5px] px-2 py-1 border rounded bg-white text-xs whitespace-nowrap text-center"
               placeholder="Text" bind:value={draft.text}
               on:input={updateDraftSize}
               on:keydown={(e) => { if (e.key === 'Enter') { e.preventDefault(); confirmDraft() } else if (e.key === 'Escape') { e.preventDefault(); cancelDraft() } }}
               on:mousedown|stopPropagation on:click|stopPropagation />
      </div>
    </div>
  {/if}
</div>

<style>
</style>
