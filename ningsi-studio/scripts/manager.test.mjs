import test from 'node:test';
import assert from 'node:assert/strict';
import { SessionDirectory, initialSession, progressPercent } from '../web/neurofocus/manager.mjs';
import { StudioAdapter } from '../web/neurofocus/studio-adapter.mjs';

test('an explicit participant remains selected when it ends or is not in the recent page', () => {
  assert.equal(initialSession([{ uuid: 'new' }], 'chosen'), 'chosen');
  assert.equal(initialSession([{ uuid: 'new' }]), 'new');
  assert.equal(initialSession([]), '');
  for (const [value, expected] of [[.42, 42], [1.5, 100], [-.2, 0], [null, 0]]) assert.equal(progressPercent(value), expected);
});

test('active discovery uses a separate status query and every request is read-only', async () => {
  const calls = [], updates = [];
  const client = new StudioAdapter(() => {}, { token: 'test-token', fetcher: async (url, options) => {
    calls.push({ url, options });
    return { ok: true, headers: new Headers(), json: async () => ({ items: [{ uuid: url.includes('status=running') ? 'active' : 'recent' }] }) };
  } });
  const directory = new SessionDirectory(client, { onUpdate: update => updates.push(update) });
  await directory.refresh();
  assert.equal(updates[0].active[0].uuid, 'active');
  assert.equal(updates[0].recent[0].uuid, 'recent');
  assert.equal(calls.length, 2);
  assert.ok(calls.some(({ url }) => url.startsWith('/api/sessions?') && url.includes('status=running')));
  assert.ok(calls.some(({ url }) => url.startsWith('/api/test-history?') && url.includes('limit=8')));
  assert.ok(calls.every(({ options }) => !options.method || options.method === 'GET'));
  assert.ok(calls.every(({ options }) => options.headers['X-API-Token'] === 'test-token'));
});

test('background refresh does not overlap and stopping aborts and ignores old results', async () => {
  const deferred = [], signals = [], updates = [];
  const directory = new SessionDirectory({ list: ({ signal }) => {
    signals.push(signal); return new Promise(resolve => deferred.push(resolve));
  } }, { onUpdate: update => updates.push(update) });
  const first = directory.refresh();
  assert.equal(directory.refresh(), first);
  assert.equal(deferred.length, 2);
  directory.stop();
  assert.ok(signals.every(signal => signal.aborted));
  deferred[0]([{ uuid: 'old' }]); deferred[1]([]); await first;
  assert.equal(updates.length, 0);
  const next = directory.refresh();
  deferred[2]([{ uuid: 'fresh' }]); deferred[3]([]); await next;
  assert.equal(updates[0].active[0].uuid, 'fresh');
});

test('an unavailable server reports an error and a later refresh recovers', async () => {
  const errors = [], updates = []; let fail = true;
  const directory = new SessionDirectory({ list: async () => {
    if (fail) throw new Error('offline'); return [];
  } }, { onUpdate: update => updates.push(update), onError: error => errors.push(error.message) });
  await directory.refresh(); assert.deepEqual(errors, ['offline']);
  fail = false; await directory.refresh(); assert.equal(updates.length, 1);
});

test('manager deep links preserve token and session without replacing the current Studio URL', async () => {
  globalThis.window = { location: { origin: 'http://localhost:8765', search: '?token=local-token', href: 'http://localhost:8765/#/flow' } };
  globalThis.document = { querySelector: () => null };
  const { managerURL } = await import('../web/js/manager-link.js');
  const url = new URL(managerURL('a'.repeat(32)));
  assert.equal(url.pathname, '/neurofocus/');
  assert.equal(url.searchParams.get('session'), 'a'.repeat(32));
  assert.equal(url.searchParams.get('token'), 'local-token');
  assert.equal(window.location.href, 'http://localhost:8765/#/flow');
});
