/** Open the observer independently, keeping the participant's Studio page alive. */
export function managerURL(session = '') {
  const target = new URL('/neurofocus/', window.location.origin);
  if (session) target.searchParams.set('session', session);
  const token = document.querySelector('meta[name="ningsi-api-token"]')?.content
    || new URLSearchParams(window.location.search).get('token');
  if (token) target.searchParams.set('token', token);
  return target.href;
}
