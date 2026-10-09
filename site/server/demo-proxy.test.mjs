import assert from 'node:assert/strict';
import test from 'node:test';
import { proxyDemo } from './demo-proxy.mjs';

const env = {DEMO_ENABLED:'true',SITE_URL:'https://site.example',DEMO_API_ORIGIN:'https://demo.example'};
const context = (path,init={}) => ({env,request:new Request('https://site.example'+path,init)});
const json = (value,status=200,headers={}) => new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json',...headers}});
const health = {model_loaded:true,method:'classical_v2',features:85};

test('health uses the configured fixed upstream and returns secure same-origin JSON',async()=>{
  const response=await proxyDemo(context('/api/healthz'),async(url,options)=>{
    assert.equal(url,'https://demo.example/healthz');
    assert.equal(options.method,'GET');
    assert.equal(options.headers.Origin,'https://site.example');
    assert.equal(options.redirect,'manual');
    return json(health);
  });
  assert.equal(response.status,200);
  assert.deepEqual(await response.json(),health);
  assert.equal(response.headers.get('Cache-Control'),'no-store');
  assert.equal(response.headers.get('X-Content-Type-Options'),'nosniff');
  assert(!response.headers.has('Access-Control-Allow-Origin'));
});

test('upload bytes survive forwarding, while visitor credentials do not',async()=>{
  const multipart='--test\r\nContent-Disposition: form-data; name="file"; filename="test.jpg"\r\n\r\nbytes\r\n--test--\r\n';
  const response=await proxyDemo(context('/api/detect',{method:'POST',body:multipart,headers:{
    'Content-Type':'multipart/form-data; boundary=test',Origin:env.SITE_URL,Cookie:'private=secret',Authorization:'Bearer secret'
  }}),async(url,options)=>{
    assert.equal(url,'https://demo.example/api/detect');
    assert.equal(new TextDecoder().decode(options.body),multipart);
    assert(!options.headers.Cookie && !options.headers.Authorization);
    return json({label:'Real',probability_ai:0.4});
  });
  assert.equal(response.status,200);
  assert.equal((await response.json()).probability_ai,0.4);
});

test('unsupported paths, methods, origins, and media types never reach upstream',async()=>{
  const never=()=>{throw new Error('Unexpected upstream request');};
  for(const [request,status] of [
    [context('/api/other'),404],
    [context('/api/detect'),405],
    [context('/api/detect',{method:'POST',headers:{Origin:'https://other.example'}}),403],
    [context('/api/detect',{method:'POST',body:'text'}),400]
  ]) assert.equal((await proxyDemo(request,never)).status,status);
});

test('disabled or invalid configuration never contacts an upstream',async()=>{
  for(const invalid of [{...env,DEMO_ENABLED:'false'},{...env,DEMO_API_ORIGIN:'http://demo.example'},{...env,DEMO_API_ORIGIN:'https://demo.example/path'},{...env,DEMO_API_ORIGIN:'https://user:secret@demo.example'}]) {
    const response=await proxyDemo({...context('/api/healthz'),env:invalid},()=>{throw new Error('Unexpected fetch');});
    assert.equal(response.status,503);
  }
});

test('oversized multipart bodies are rejected before forwarding',async()=>{
  const response=await proxyDemo(context('/api/detect',{method:'POST',body:new Uint8Array(1048576+8193),headers:{'Content-Type':'multipart/form-data; boundary=test'}}),()=>{throw new Error('Unexpected fetch');});
  assert.equal(response.status,413);
  assert.equal((await response.json()).detail,'too_large');
});

test('cold-start HTML, redirects, network errors, and incorrect models produce retryable JSON',async()=>{
  for(const upstream of [
    ()=>new Response('<html>Starting</html>',{headers:{'Content-Type':'text/html'}}),
    ()=>new Response(null,{status:302,headers:{Location:'https://other.example'}}),
    ()=>{throw new TypeError('Network failure');},
    ()=>json({...health,method:'wrong_model'}),
    ()=>json({detail:'busy'},503)
  ]) {
    const response=await proxyDemo(context('/api/healthz'),upstream);
    assert.equal(response.status,503);
    assert.equal((await response.json()).detail,'waking');
  }
});

test('rate-limit response and retry delay are preserved without replaying the upload',async()=>{
  let requests=0;
  const response=await proxyDemo(context('/api/detect',{method:'POST',body:'body',headers:{'Content-Type':'multipart/form-data; boundary=test'}}),()=>{
    requests++;return json({detail:'rate_limit'},429,{'Retry-After':'60'});
  });
  assert.equal(requests,1);
  assert.equal(response.status,429);
  assert.equal(response.headers.get('Retry-After'),'60');
});

test('invalid or oversized upstream JSON is not sent to the browser',async()=>{
  for(const contents of ['not JSON','x'.repeat(128*1024+1)]) {
    const response=await proxyDemo(context('/api/healthz'),()=>new Response(contents,{headers:{'Content-Type':'application/json'}}));
    assert.equal(response.status,503);
    assert.equal((await response.json()).detail,'waking');
  }
});
