import assert from 'node:assert/strict';
import { readFile, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEMO_ENABLED } from './config.mjs';

const site = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(site);
const out = path.resolve(process.env.OUTPUT_DIR || path.join(site,'dist'));
const evaluation = JSON.parse(await readFile(path.join(repo,'reports/eval_v2_20260615.json'),'utf8'));
const homepage = await readFile(path.join(out,'index.html'),'utf8');
const testPage = await readFile(path.join(out,'test/index.html'),'utf8');
if (DEMO_ENABLED) {
  assert(homepage.includes('/test/'));
  assert(testPage.includes('id="demo-form"'));
  assert(testPage.includes('This free demo runs the 85-feature classical v2 model.'));
} else {
  assert(homepage.includes('Live demo offline, run it locally'));
  assert(!/<input\b[^>]*type=["']file/i.test(testPage));
}
assert(!/<input\b[^>]*type=["']file/i.test(homepage),'An offline demo must not render an upload input.');
assert(!/<form\b/i.test(homepage),'The homepage should link to the dedicated test page.');
assert(homepage.includes((evaluation.unified_overall.accuracy*100).toFixed(1)+'%'));
assert(homepage.includes(evaluation.unified_overall.auc.toFixed(3)));
assert(homepage.includes('Three acceptance gates remain unmet.'));
const readme = await readFile(path.join(repo,'README.md'),'utf8');
const sourceCommands = readme.match(/## Quick Start\s+```bash\n([\s\S]*?)```/)[1].trim();
const decode = text => text.replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');
const renderedCommands = decode(homepage.match(/<code id="start-commands">([\s\S]*?)<\/code>/)[1]);
assert(renderedCommands.endsWith(sourceCommands),'README laptop commands must be copied verbatim.');
let count=0,total=0;
async function inspect(directory) {
  for(const entry of await readdir(directory,{withFileTypes:true})) {
    const full = path.join(directory,entry.name);
    if(entry.isDirectory()) await inspect(full);
    else {
      count++;
      const bytes=(await stat(full)).size; total+=bytes;
      assert(bytes<=25*1024*1024,`Pages file limit: ${full}`);
      assert(!/\.(py|pkl|pyc)$/.test(full),'Python and model bundles must not be deployed.');
      if(full.endsWith('.json')) JSON.parse(await readFile(full,'utf8'));
      if(full.endsWith('.html')) {
        const html=await readFile(full,'utf8');
        assert.equal((html.match(/<h1\b/g)||[]).length,1,'One page heading is required.');
        assert(!/\b(?:src|href)=["']http:\/\//.test(html),'Mixed-content URLs are not allowed.');
        assert(!/<(?:style|script)[^>]*>\s*(?:@import|document\.)/.test(html),'Keep behavior and styles in hashed assets.');
        for(const [,href] of html.matchAll(/\b(?:href|src)="([^"#]+)"/g)) {
          if(!href.startsWith('/')) continue;
          const target=href.split('#')[0];
          const resolved=path.join(out,target.endsWith('/')?target+'index.html':target);
          assert((await stat(resolved)).isFile(),`Missing linked asset: ${href}`);
        }
        JSON.parse(html.match(/<script type="application\/ld\+json">(.*?)<\/script>/s)[1]);
      }
    }
  }
}
await inspect(out);
assert(count<=20000,'Pages free-plan file count exceeded.');
for(const [source,target] of [['reports/eval_v2_20260615.json','data/evaluation.json'],['experiment_v1.json','data/experiment.json'],['reports/family_analysis_20260615.json','data/family-analysis.json']]) {
  assert((await readFile(path.join(repo,source))).equals(await readFile(path.join(out,target))),`Evidence was changed: ${source}`);
}
console.log(`PASS: ${count} files / ${total} bytes; routes, links, evidence copies, README commands, disabled demo, metadata, Pages limits.`);
