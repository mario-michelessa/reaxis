<script>
  import { createEventDispatcher } from 'svelte'

  export let systemName = 'Reaxis'
  export let busy = false
  export let error = ''

  const dispatch = createEventDispatcher()
  let sessionName = ''

  function submit() {
    const value = String(sessionName || '').trim()
    if (!value || busy) return
    dispatch('submit', { sessionName: value })
  }
</script>

<div class="loading-gate-shell">
  <div class="loading-gate-card">
    <div class="loading-gate-brand-name">{systemName}</div>
    <div class="loading-gate-brand-subtitle">enter name of the session</div>
    <input
      id="session-gate-input"
      class="loading-gate-input"
      type="text"
      bind:value={sessionName}
      placeholder="session_name"
      aria-label="Session name"
      autocomplete="off"
      autocapitalize="off"
      spellcheck="false"
      disabled={busy}
      on:keydown={(e) => {
        if (e.key !== 'Enter') return
        try { e.preventDefault() } catch (_) {}
        submit()
      }}
    />

    {#if error}
      <div class="loading-gate-error">{error}</div>
    {/if}

    <button
      type="button"
      class="loading-gate-submit"
      disabled={busy || !String(sessionName || '').trim()}
      on:click={submit}
    >
      {#if busy}Opening...{:else}Enter{/if}
    </button>
  </div>
</div>

<style>
  .loading-gate-shell {
    min-height: 100vh;
    display: grid;
    place-items: center;
    padding: 32px;
    background: #ffffff;
  }

  .loading-gate-card {
    width: min(100%, 360px);
    padding: 20px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
    display: grid;
    gap: 12px;
  }

  .loading-gate-brand-name {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #2563eb;
    line-height: 1;
  }

  .loading-gate-brand-subtitle {
    font-size: 14px;
    color: #64748b;
  }

  .loading-gate-input {
    width: 100%;
    padding: 12px 13px;
    border-radius: 12px;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #0f172a;
    font-size: 15px;
    outline: none;
  }

  .loading-gate-input:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
  }

  .loading-gate-error {
    padding: 10px 12px;
    border-radius: 12px;
    background: #fff1f2;
    border: 1px solid #fecdd3;
    color: #be123c;
    font-size: 13px;
  }

  .loading-gate-submit {
    padding: 12px 14px;
    border: none;
    border-radius: 12px;
    background: #2563eb;
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
  }

  .loading-gate-submit:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
</style>
