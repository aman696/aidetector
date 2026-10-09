import { readFile, writeFile, mkdir, readdir, stat, rm, copyFile, realpath } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { createContent } from './content.mjs';
import { renderPage } from './render.mjs';
import { DEMO_ENABLED, DEMO_API_ORIGIN } from './config.mjs';

const site = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(site);
const out = path.resolve(process.env.OUTPUT_DIR || path.join(site, 'dist'));
if (out === repo || out === site || out === path.parse(out).root) throw new Error('Unsafe output directory.');
const siteUrl = (process.env.SITE_URL || '').replace(/\/$/, '');
if (!siteUrl) throw new Error('Set SITE_URL to the production origin before building.');
const url = new URL(siteUrl);
if (url.pathname !== '/' || url.search || url.hash || url.username || url.password || url.protocol !== 'https:') {
  throw new Error('SITE_URL must be an HTTPS origin without a path, query, or credentials.');
}
if (DEMO_ENABLED) {
  const api = new URL(DEMO_API_ORIGIN);
  if (api.protocol !== 'https:' || api.pathname !== '/' || api.search || api.hash || api.username || api.password) throw new Error('An enabled demo requires a real HTTPS DEMO_API_ORIGIN.');
  const health = await fetch(api.origin + '/healthz', { signal:AbortSignal.timeout(120000) });
  const status = await health.json();
  if (!health.ok || !status.model_loaded || status.method !== 'classical_v2' || status.features !== 85) throw new Error('The hosted classical detector is not ready.');
}
const evaluation = JSON.parse(await readFile(path.join(repo, 'reports/eval_v2_20260615.json'), 'utf8'));
const experiment = JSON.parse(await readFile(path.join(repo, 'experiment_v1.json'), 'utf8'));
const readme = await readFile(path.join(repo, 'README.md'), 'utf8');
const commands = readme.match(/## Quick Start\s+```bash\n([\s\S]*?)```/)?.[1].trim();
if (!commands) throw new Error('README Quick Start was not found.');
const clone = readme.match(/git clone [^\n]+\ncd [^\n]+/)?.[0];
if (!clone) throw new Error('README clone commands were not found.');
const c = createContent(evaluation, experiment, `${clone}\n\n${commands}`, { demoEnabled:DEMO_ENABLED });
await rm(out, { recursive: true, force: true });
await mkdir(path.join(out, 'assets'), { recursive: true });
const digest = text => createHash('sha256').update(text).digest('hex').slice(0, 12);
const assets = {};
for (const [key, filename, extension] of [['css','styles.css','css'],['js','client.js','js'],['theme','theme.js','js'],['demo','demo.js','js']]) {
  const source = await readFile(path.join(site, filename));
  const target = `/assets/${key}.${digest(source)}.${extension}`;
  assets[key] = target;
  await writeFile(path.join(out, target), source);
}
async function copyPublic(from, to) {
  await mkdir(to, { recursive: true });
  for (const entry of await readdir(from, { withFileTypes: true })) {
    const source = path.join(from, entry.name), target = path.join(to, entry.name);
    if (entry.isDirectory()) await copyPublic(source, target);
    else {
      const resolved = await realpath(source);
      if (!resolved.startsWith(repo + path.sep)) throw new Error(`Public asset escapes repository: ${source}`);
      await copyFile(source, target);
    }
  }
}
await copyPublic(path.join(site, 'public'), out);
const schemaHashes = [];
for (const page of ['home','resources','test','404']) {
  const result = renderPage(c, { page, siteUrl, assets, demoEnabled:DEMO_ENABLED, demoApiOrigin:DEMO_API_ORIGIN });
  const relative = page === 'home' ? 'index.html' : page === 'resources' ? 'resources/index.html' : page === 'test' ? 'test/index.html' : '404.html';
  await mkdir(path.dirname(path.join(out, relative)), { recursive:true });
  await writeFile(path.join(out, relative), result.html);
  schemaHashes.push(`'sha256-${createHash('sha256').update(result.schema).digest('base64')}'`);
}
const models = [];
for (const [key, record] of Object.entries(experiment.model)) {
  if (!record.file) continue;
  const modelPath = path.join(repo, record.file), bytes = await readFile(modelPath);
  const hash = createHash('sha256').update(bytes).digest('hex');
  if (hash !== record.sha256) throw new Error(`Model hash differs from experiment record: ${record.file}`);
  models.push({ name:key, file:record.file, sha256:hash, bytes:bytes.length, features:record.n_features, pages_asset_limit_bytes:25*1024*1024, exceeds_pages_asset_limit:bytes.length>25*1024*1024, included_in_site:false });
}
await writeFile(path.join(out,'data/model-metadata.json'),JSON.stringify({source:'experiment_v1.json',models},null,2)+'\n');
await writeFile(path.join(out,'data/api.json'),JSON.stringify({note:c.api.note,framework:'FastAPI',entry_point:'app.py',hosted_entry_point:DEMO_ENABLED?'demo/api.py':null,runtime:DEMO_ENABLED?'Python 3.12.12 (hosted); see README for local setup':'Python 3.10+',demo_enabled:DEMO_ENABLED,hosted_demo:DEMO_ENABLED?siteUrl+'/test/':null,hosted_api_origin:DEMO_ENABLED?DEMO_API_ORIGIN:null,hosted_model:DEMO_ENABLED?'classical_v2':null,local_detect_route:{method:'POST',path:'/api/detect'},static_site_has_inference_api:false,same_origin_proxy:{health:'/api/healthz',detect:'/api/detect'},pages_functions:['functions/api/[route].js'],incompatibility:c.api.incompatibility,options:c.api.alternatives},null,2)+'\n');
await writeFile(path.join(out,'robots.txt'),`User-agent: *\nAllow: /\nSitemap: ${siteUrl}/sitemap.xml\n`);
await writeFile(path.join(out,'sitemap.xml'),`<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>${siteUrl}/</loc></url><url><loc>${siteUrl}/resources/</loc></url><url><loc>${siteUrl}/test/</loc></url></urlset>\n`);
await writeFile(path.join(out,'llms.txt'),`# ${c.brand}\n\n${c.description}\n\n${c.hero.note}\n\n## Resources\n\n${c.resources.entries.map(e=>`- [${e.label}](${siteUrl}${e.href}): ${e.description}`).join('\n')}\n`);
const csp = `default-src 'none'; script-src 'self' ${[...new Set(schemaHashes)].join(' ')}; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'; upgrade-insecure-requests`;
await writeFile(path.join(out,'_headers'),`/*\n  Content-Security-Policy: ${csp}\n  X-Content-Type-Options: nosniff\n  X-Frame-Options: DENY\n  Referrer-Policy: strict-origin-when-cross-origin\n  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()\n  Strict-Transport-Security: max-age=31536000\n\n/\n  Cache-Control: public, max-age=0, must-revalidate, no-transform\n\n/resources\n  Cache-Control: public, max-age=0, must-revalidate, no-transform\n\n/resources/\n  Cache-Control: public, max-age=0, must-revalidate, no-transform\n\n/test\n  Cache-Control: public, max-age=0, must-revalidate, no-transform\n\n/test/\n  Cache-Control: public, max-age=0, must-revalidate, no-transform\n\n/404\n  Cache-Control: public, max-age=0, must-revalidate, no-transform\n\n/404.html\n  Cache-Control: public, max-age=0, must-revalidate, no-transform\n\n/assets/*\n  ! Content-Security-Policy\n  Cache-Control: public, max-age=31536000, immutable\n\n/data/*\n  Content-Type: application/json; charset=utf-8\n\n/docs/*.md\n  Content-Type: text/markdown; charset=utf-8\n\n/sitemap.xml\n  Content-Type: application/xml; charset=utf-8\n\n/docs/LICENSE.txt\n  Content-Type: text/plain; charset=utf-8\n`);
await writeFile(path.join(out,'_routes.json'),JSON.stringify({version:1,include:['/api/healthz','/api/detect'],exclude:[]},null,2)+'\n');
const deployed = [];
async function collect(directory) {
  for (const entry of await readdir(directory, { withFileTypes:true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) await collect(full);
    else {
      const size = (await stat(full)).size;
      if (size > 25*1024*1024) throw new Error(`Asset exceeds Pages 25 MiB limit: ${full}`);
      deployed.push({path:'/'+path.relative(out,full).split(path.sep).join('/'),bytes:size});
    }
  }
}
await collect(out);
if (deployed.length > 20000) throw new Error('Pages free-plan file limit exceeded.');
const total = deployed.reduce((sum,file)=>sum+file.bytes,0);
console.log(`Built ${deployed.length} static files (${total.toLocaleString('en-US')} bytes) to ${out}`);
console.log(`DEMO_ENABLED=${DEMO_ENABLED}; Python/model bundles excluded. Pages Functions proxy /api/healthz and /api/detect only.`);
