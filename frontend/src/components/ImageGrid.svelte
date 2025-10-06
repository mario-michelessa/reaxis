<script>
  export let items = [];
  export let selected = new Set();
  export let onToggleSelect = (id) => {};
  export let onClick = (item) => {};
  // Column sizing is handled inline for broad compatibility
  const gridStyle = 'grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));';
</script>

<div class="grid gap-[1px] w-full" style={gridStyle}>
  {#each items as item}
    <div class="relative group overflow-hidden cursor-pointer bg-white" role="button" tabindex="0" on:click={() => onClick(item)} on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && onClick(item)}>
      <img class="w-full h-44 object-cover" alt={item.label} src={item.url} loading="lazy" />
      <!-- Top-right checkbox: visible on hover, always visible when selected -->
      <input
        type="checkbox"
        class="absolute top-1 right-1 w-5 h-5 opacity-0 group-hover:opacity-100 transition-opacity"
        class:opacity-100={selected.has(item.id)}
        checked={selected.has(item.id)}
        on:click|stopPropagation={() => onToggleSelect(item.id)}
      />
      <!-- Bottom-right tag with class name -->
      <div class="absolute bottom-1 right-1 chip text-xs">
        {item.className}
      </div>
    </div>
  {/each}
</div>
