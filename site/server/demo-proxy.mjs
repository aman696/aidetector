const MAX_BODY = 1048576 + 8192;
const MAX_RESPONSE = 128 * 1024;
const securityHeaders = {
  'Content-Type':'application/json; charset=utf-8',
  'Cache-Control':'no-store',
  'X-Content-Type-Options':'nosniff',
  'X-Frame-Options':'DENY',
  'Strict-Transport-Security':'max-age=31536000',
  'Permissions-Policy':'camera=(), microphone=(), geolocation=(), payment=()',
  'Referrer-Policy':'no-referrer',
  'Content-Security-Policy':"default-src 'none'; frame-ancestors 'none'"
};
const error = (detail,status,extra={}) => new Response(JSON.stringify({detail}),{status,headers:{...securityHeaders,...extra}});

async function boundedBody(stream,limit) {
  if (!stream) return new Uint8Array();
  const reader = stream.getReader(), chunks = [];
  let length = 0;
  try {
    while (true) {
      const {value,done} = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > limit) { await reader.cancel(); throw new RangeError('too_large'); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk,offset); offset += chunk.length; }
  return bytes;
}

export async function proxyDemo({request,env},fetchUpstream=fetch) {
  const pathname = new URL(request.url).pathname.replace(/\/$/,'');
  const isHealth = pathname === '/api/healthz';
  if (!isHealth && pathname !== '/api/detect') return error('not_found',404);
  const method = isHealth ? 'GET' : 'POST';
  if (request.method !== method) return error('method_not_allowed',405,{Allow:method});
  if (env.DEMO_ENABLED !== 'true') return error('offline',503);
  let upstream,site;
  try {
    upstream = new URL(env.DEMO_API_ORIGIN);
    site = new URL(env.SITE_URL);
    for (const url of [upstream,site]) {
      if (url.protocol !== 'https:' || url.pathname !== '/' || url.search || url.hash || url.username || url.password) throw new Error();
    }
  } catch { return error('offline',503); }
  const origin = request.headers.get('Origin');
  if (origin && origin !== site.origin) return error('origin',403);
  const headers = {Accept:'application/json',Origin:site.origin};
  let body;
  if (!isHealth) {
    const type = request.headers.get('Content-Type') || '';
    if (!/^multipart\/form-data;\s*boundary=/i.test(type)) return error('invalid_type',400);
    if (Number(request.headers.get('Content-Length')) > MAX_BODY) return error('too_large',413);
    try { body = await boundedBody(request.body,MAX_BODY); }
    catch { return error('too_large',413); }
    headers['Content-Type'] = type;
  }
  try {
    // A fixed upstream and two fixed paths prevent this from becoming an open proxy.
    const response = await fetchUpstream(upstream.origin + (isHealth ? '/healthz' : '/api/detect'),{
      method,headers,body,redirect:'manual',signal:AbortSignal.timeout(isHealth ? 25000 : 150000)
    });
    if (!(response.headers.get('Content-Type') || '').includes('application/json')) return error(isHealth ? 'waking' : 'offline',503);
    const bytes = await boundedBody(response.body,MAX_RESPONSE);
    const result = JSON.parse(new TextDecoder().decode(bytes));
    if (isHealth && (!response.ok || !result.model_loaded || result.method !== 'classical_v2' || result.features !== 85)) return error('waking',503);
    const extra = {};
    const retry = response.headers.get('Retry-After');
    if (retry && /^\d+$/.test(retry)) extra['Retry-After'] = retry;
    return new Response(bytes,{status:response.status,headers:{...securityHeaders,...extra}});
  } catch { return error(isHealth ? 'waking' : 'offline',503); }
}
