<script>
  export let name = ''
  export let size = 18
  export let filled = true
  export let weight = 400
  export let grade = 0
  export let optical = 24

  $: sizeCss = typeof size === 'number' ? `${size}px` : String(size || '18px')
</script>

<span
  class="material-symbols-rounded material-symbol"
  aria-hidden="true"
  style={`font-size:${sizeCss};font-variation-settings:'FILL' ${filled ? 1 : 0}, 'wght' ${weight}, 'GRAD' ${grade}, 'opsz' ${optical};`}
>{name}</span>

