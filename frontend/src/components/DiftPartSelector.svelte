<script>
  import { createEventDispatcher } from 'svelte'
  export let selected = '' // '11'..'33'
  const dispatch = createEventDispatcher()

  const rows = [0,1,2,]
  const cols = [0,1,2]

  function pick(r, c) {
    const part = `${r}${c}`
    selected = part
    dispatch('select', { part })
  }
</script>

<div class="grid grid-cols-3 gap-1" style="width: 114px;">
  {#each rows as r}
    {#each cols as c}
      {#key `${r}${c}`}
        <button
          class="w-9 h-9 border rounded text-xs"
          class:bg-blue-600={selected === `${r}${c}`}
          class:text-white={selected === `${r}${c}`}
          on:click={() => pick(r, c)}
        >{r}{c}</button>
      {/key}
    {/each}
  {/each}
  </div>

<style>
</style>

