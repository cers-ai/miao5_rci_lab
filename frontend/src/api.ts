export async function api<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const res = await fetch('/api' + path, { ...options, headers, credentials: 'same-origin' });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `请求失败 (${res.status})` }));
    if (res.status === 401 && path !== '/login') window.dispatchEvent(new Event('session-expired'));
    const detail = Array.isArray(body.detail) ? body.detail.map((e: any) => `${e.loc?.slice(1).join('.')}: ${e.msg}`).join('；') : body.detail;
    throw new Error(detail || `请求失败 (${res.status})`);
  }
  return res.json();
}
export const post = (path: string, body: any = {}) => api(path, { method: 'POST', body: JSON.stringify(body) });
export const put = (path: string, body: any) => api(path, { method: 'PUT', body: JSON.stringify(body) });
export const seconds = (ms: number | null | undefined) => ms == null ? '—' : `${(ms / 1000).toFixed(2)} 秒`;
export const score = (v: number | null | undefined) => v == null ? '—' : v.toFixed(3);
export const date = (s?: string) => s ? new Date(s).toLocaleString('zh-CN', { hour12: false }) : '—';
