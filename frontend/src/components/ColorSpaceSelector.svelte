<script>
  import { createEventDispatcher } from 'svelte'
  const dispatch = createEventDispatcher()
  let selected = 'color_rgb'

  const options = [
    { value: 'color_rgb', label: 'RGB' },
    { value: 'color_lch', label: 'LCh (CIELAB)' },
    { value: 'color_hsv', label: 'HSV (hue/sat)' },
  ]

  function pick(value) {
    selected = value
    dispatch('change', value)
  }
</script>

<div class="grid gap-1">
  {#each options as opt}
    <label class="inline-flex items-center gap-2 text-base ml-2">
      <input type="radio" name="color-space" value={opt.value} checked={selected===opt.value} on:change={(e)=>pick(e.currentTarget.value)} />
      <span>{opt.label}</span>
    </label>
  {/each}
</div>

<style>
</style>

