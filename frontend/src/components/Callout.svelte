<script>
  export let title = ''
  let hidden = false
  function hide() {
    hidden = true
  }
</script>

{#if !hidden}
<div class="callout">
  <button class="callout-close btn btn-xs btn-ui-secondary" on:click={hide} aria-label="Hide">✕</button>
  <div style="text-align:left">
    {#if title}
      <div class="callout-title">{title}</div>
    {/if}
    <div class="callout-body"><slot /></div>
  </div>
</div>
{/if}

<style>
</style>
