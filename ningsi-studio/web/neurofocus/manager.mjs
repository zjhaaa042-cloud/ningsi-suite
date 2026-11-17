// The directory stays connected while another tab runs the participant workflow.
// All requests are reads; selecting a session never starts an experiment.
export class SessionDirectory {
  constructor(client, { onUpdate = () => {}, onError = () => {}, intervalMs = 5000 } = {}) {
    Object.assign(this, { client, onUpdate, onError, intervalMs });
    this.timer = null; this.pending = null; this.generation = 0;
  }
  refresh() {
    if (this.pending) return this.pending;
    const generation = this.generation;
    this.controller = new AbortController();
    const signal = this.controller.signal;
    this.pending = (async () => {
      try {
        // Query active sessions separately so newer history cannot hide them.
        const [active, recent] = await Promise.all([
          this.client.list({ status: 'running', limit: 100, signal }),
          this.client.list({ limit: 8, signal }),
        ]);
        if (generation === this.generation) this.onUpdate({ active, recent, updatedAt: Date.now() });
      } catch (error) {
        if (generation === this.generation && !signal.aborted) this.onError(error);
      } finally {
        if (generation === this.generation) this.pending = null;
      }
    })();
    return this.pending;
  }
  start() {
    if (this.timer !== null) return;
    this.timer = setInterval(() => this.refresh(), this.intervalMs);
    return this.refresh();
  }
  stop() {
    this.generation++; this.controller?.abort();
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null; this.pending = null;
  }
}

export function initialSession(active, requested = '') {
  // An explicit link must never silently attach to a different participant.
  return requested || active[0]?.uuid || '';
}

export function progressPercent(value) {
  return Number.isFinite(value) ? Math.round(Math.max(0, Math.min(1, value)) * 100) : 0;
}
