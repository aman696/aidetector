// Run against the deployed origin: node verify.mjs https://your-domain
import assert from 'node:assert/strict';
import { createContent } from './content.mjs';

const origin = (process.argv[2] || '').replace(/\/$/,'');
if(!origin) throw new Error('Supply the origin to verify.');
const read = async route => {
  const response=await fetch(origin+route,{redirect:'follow',signal:AbortSignal.timeout(20000)});
  assert.equal(response.status,200,`${route}: HTTP ${response.status}`);
  return response;
};
const evaluation=await (await read('/data/evaluation.json')).json();
const experiment=await (await read('/data/experiment.json')).json();
const c=createContent(evaluation,experiment,'');
const routes=[['/','text/html'],['/resources/','text/html'],['/test/','text/html'],['/robots.txt','text/plain'],['/sitemap.xml','application/xml'],['/llms.txt','text/plain'],['/favicon.svg','image/svg+xml'],['/opengraph.png','image/png'],...c.resources.entries.map(item=>[item.href,item.type==='JSON'?'application/json':item.type==='MARKDOWN'?'text/markdown':'text/plain'])];
const results=[];
for(const [route,type] of routes) {
  const response=await read(route), actual=response.headers.get('content-type')||'';
  assert(actual.includes(type),`${route}: expected ${type}, received ${actual}`);
  assert.equal(response.headers.get('x-content-type-options'),'nosniff',`${route}: missing nosniff header`);
  const text=type==='image/png'?'':await response.text();
  if(type==='application/json') JSON.parse(text);
  if(type==='text/html') {
    assert(!/\b(?:src|href)=["']http:\/\//.test(text),`${route}: mixed content`);
    assert(response.headers.get('content-security-policy'),`${route}: missing CSP`);
    for(const [,asset] of text.matchAll(/(?:src|href)="(\/assets\/[^"]+)"/g)) {
      if(results.some(item=>item.route===asset)) continue;
      const file=await read(asset), assetType=asset.endsWith('.css')?'text/css':/javascript/;
      assert(typeof assetType==='string'?(file.headers.get('content-type')||'').includes(assetType):assetType.test(file.headers.get('content-type')||''),`${asset}: wrong content type`);
      assert((file.headers.get('cache-control')||'').includes('immutable'),`${asset}: missing immutable caching`);
      results.push({route:asset,status:file.status,content_type:file.headers.get('content-type')});
    }
  }
  results.push({route,status:response.status,content_type:actual});
}
for(const route of ['/this-route-does-not-exist','/api/detect']) {
  const response=await fetch(origin+route,{method:route==='/api/detect'?'POST':'GET',signal:AbortSignal.timeout(20000)});
  const expected=route==='/api/detect'?[404,405]:[404];
  assert(expected.includes(response.status),`${route}: expected ${expected.join(' or ')}, received ${response.status}`);
  results.push({route,status:response.status,expected});
}
console.log(JSON.stringify({origin,verified_at:new Date().toISOString(),passed:true,results},null,2));
