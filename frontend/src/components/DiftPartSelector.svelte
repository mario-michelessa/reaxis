<script>
  import { createEventDispatcher } from 'svelte'
  let selected = '' // '11'..'33'
  let method = '' // dino or dift_sd_partXY
  const dispatch = createEventDispatcher()

  const rows = [0, 1, 2]
  const cols = [0, 1, 2]

  function pick(r, c) {
    // partXY with X=row, Y=column (dataset expects `${r}${c}`)
    const part = `${r}${c}`
    if (selected === part) {
      method = 'dino'
      selected = ''
    } else {
      method = 'dift_sd_part' + part
      selected = part
    }
    dispatch('change', method)
  }

  function symbolForPart(r, c) {
    const symbols = [  
      ['↖','↑','↗'],
      ['←','·','→'],
      ['↙','↓','↘'],
    ]
    return symbols[r][c]
  }
</script>
<div class="flex justify-center ">
<div class="grid grid-cols-3 gap-0" style="width: 114px;">
  {#each rows as r}
    {#each cols as c}
      {#key `${r}${c}`}
        <button
          class="w-9 h-9 border rounded text-xs"
          class:bg-gray-200={selected !== `${r}${c}`}
          class:bg-blue-600={selected === `${r}${c}`}
          on:click={() => pick(r, c)}
        > {symbolForPart(r, c)}</button>
      {/key}
    {/each}
  {/each}
  </div>
</div>
<style>
</style>
