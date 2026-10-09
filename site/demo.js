const root = document.querySelector('[data-demo]');
if (root) {
  const copy = JSON.parse(root.dataset.demoCopy);
  const form = root.querySelector('form');
  const input = root.querySelector('input');
  const submit = form.querySelector('button');
  const status = root.querySelector('[data-demo-status]');
  const error = root.querySelector('[data-demo-error]');
  let previewUrl;
  let result;
  let fileIsValid = false;
  let selection = 0;
  const preview = root.querySelector('[data-preview]');
  const needsPreparation = file => file.size > 1048576 || preview.naturalWidth * preview.naturalHeight > 1048576 || Math.max(preview.naturalWidth,preview.naturalHeight) > 4096;
  const describePreparation = (width,height) => copy.prepared.replace('{width}',width).replace('{height}',height);
  const isAnimated = async file => {
    const header = new Uint8Array(await file.slice(0,32).arrayBuffer());
    if (header[0] === 137 && header[1] === 80 && header[2] === 78 && header[3] === 71) {
      let offset = 8;
      while (offset + 8 <= file.size) {
        const chunk = new Uint8Array(await file.slice(offset,offset+8).arrayBuffer());
        const type = String.fromCharCode(...chunk.slice(4));
        if (type === 'acTL') return true;
        if (type === 'IDAT' || type === 'IEND') return false;
        offset += new DataView(chunk.buffer).getUint32(0) + 12;
      }
    }
    return String.fromCharCode(...header.slice(0,4)) === 'RIFF'
      && String.fromCharCode(...header.slice(8,12)) === 'WEBP'
      && String.fromCharCode(...header.slice(12,16)) === 'VP8X' && Boolean(header[20] & 2);
  };
  const prepareUpload = async file => {
    const original = {width:preview.naturalWidth,height:preview.naturalHeight,bytes:file.size};
    if (!needsPreparation(file)) return {file,info:{original,analyzed:original,resized:false,reencoded:false,metadata_preserved:true}};
    status.textContent = copy.preparing;
    const scale = Math.min(1,1024 / Math.max(original.width,original.height));
    const width = Math.max(1,Math.floor(original.width * scale));
    const height = Math.max(1,Math.floor(original.height * scale));
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    let bitmap;
    try {
      const context = canvas.getContext('2d');
      if (!context) throw new Error('preparation_failed');
      const source = typeof createImageBitmap === 'function'
        ? (bitmap = await createImageBitmap(file,{resizeWidth:width,resizeHeight:height,resizeQuality:'high'})) : preview;
      context.fillStyle = '#fff'; context.fillRect(0,0,width,height);
      context.drawImage(source,0,0,width,height);
      for (const quality of [0.92,0.85,0.75,0.6]) {
        const blob = await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',quality));
        if (blob && blob.size <= 1048576) return {
          file:new File([blob],'image.jpg',{type:'image/jpeg'}),
          info:{original,analyzed:{width,height,bytes:blob.size},resized:width!==original.width || height!==original.height,reencoded:true,metadata_preserved:false}
        };
      }
      throw new Error('preparation_failed');
    } catch { throw new Error('preparation_failed'); }
    finally { bitmap?.close(); canvas.width = canvas.height = 0; }
  };
  const fail = code => {
    error.textContent = copy.errors[code] || copy.errors.offline;
    error.hidden = false;
    status.textContent = '';
  };
  input.addEventListener('change', async () => {
    const currentSelection = ++selection;
    fileIsValid = false;
    error.hidden = true;
    root.querySelector('[data-result]').hidden = true;
    root.querySelector('.selected-file').hidden = true;
    root.querySelector('[data-preparation]').hidden = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const file = input.files[0];
    if (!file) return;
    if (!/\.(jpe?g|png|webp)$/i.test(file.name)) return fail('invalid_type');
    try {
      const animated = await isAnimated(file);
      if (currentSelection !== selection) return;
      if (animated) return fail('animated_image');
    } catch { if (currentSelection === selection) fail('invalid_image'); return; }
    previewUrl = URL.createObjectURL(file);
    const image = preview;
    image.onload = () => {
      fileIsValid = true;
      const note = root.querySelector('[data-preparation]');
      note.textContent = needsPreparation(file) ? copy.willPrepare : copy.originalUpload;
      note.hidden = false;
      status.textContent = copy.ready;
    };
    image.onerror = () => fail('invalid_image');
    image.src = previewUrl;
    root.querySelector('[data-filename]').textContent = file.name;
    root.querySelector('.selected-file').hidden = false;
  });
  const validResult = value => value && value.method === 'classical_v2' && value.fallback === true
    && ['Real','AI-Generated'].includes(value.label)
    && Number.isFinite(value.probability_ai) && value.probability_ai >= 0 && value.probability_ai <= 1
    && Number.isFinite(value.analysis_time) && value.analysis_time >= 0
    && value.features?.length === 85
    && value.features.every(f=>typeof f.name==='string' && Number.isFinite(f.value));
  form.addEventListener('submit', async event => {
    event.preventDefault();
    error.hidden = true;
    if (!input.files[0]) return fail('empty');
    if (!fileIsValid) return;
    submit.disabled = true;
    input.disabled = true;
    submit.textContent = copy.analyzing;
    form.setAttribute('aria-busy','true');
    root.querySelector('[data-result]').hidden = true;
    const controller = new AbortController();
    const timer = setTimeout(()=>controller.abort(),180000);
    try {
      const upload = await prepareUpload(input.files[0]);
      const deadline = Date.now() + 120000;
      let ready = false;
      while (!ready) {
        status.textContent = copy.checking;
        try {
          const health = await fetch('/api/healthz',{signal:controller.signal,cache:'no-store',credentials:'omit'});
          const response = await health.json();
          ready = health.ok && response.model_loaded && response.method === 'classical_v2' && response.features === 85;
        } catch (exception) { if (controller.signal.aborted) throw exception; }
        if (!ready) {
          if (Date.now() >= deadline) throw new Error('offline');
          status.textContent = copy.waking;
          await new Promise((resolve,reject)=>{
            const abort = () => { clearTimeout(delay); reject(new DOMException('Aborted','AbortError')); };
            const delay = setTimeout(()=>{controller.signal.removeEventListener('abort',abort);resolve();},3000);
            controller.signal.addEventListener('abort',abort,{once:true});
          });
        }
      }
      status.textContent = copy.analyzing;
      const body = new FormData(); body.append('file',upload.file);
      const response = await fetch('/api/detect',{method:'POST',body,signal:controller.signal,credentials:'omit'});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail==='string'?data.detail:response.status===429?'rate_limit':'analysis_failed');
      if (!validResult(data)) throw new Error('invalid_result');
      result = {...data,input_processing:upload.info};
      root.querySelector('[data-processing]').textContent = upload.info.reencoded
        ? describePreparation(upload.info.analyzed.width,upload.info.analyzed.height) : copy.originalUpload;
      const percent = (data.probability_ai * 100).toFixed(1)+'%';
      root.querySelector('[data-probability]').textContent = percent;
      root.querySelector('[data-probability-bar]').value = data.probability_ai;
      root.querySelector('[data-probability-bar]').setAttribute('aria-valuetext',percent);
      root.querySelector('[data-verdict]').textContent = data.label === 'Real' ? copy.realLabel : copy.aiLabel;
      root.querySelector('[data-duration]').textContent = data.analysis_time.toFixed(2)+' '+copy.durationUnit;
      const groups = root.querySelector('[data-feature-groups]'); groups.replaceChildren();
      let offset = 0;
      for (const [name,count] of copy.groups) {
        const article=document.createElement('article'); article.className='feature-group';
        const title=document.createElement('h3'); title.textContent=name; article.append(title);
        const table=document.createElement('table'); table.className='feature-table';
        const head=document.createElement('thead'), row=document.createElement('tr');
        for (const label of [copy.featureName,copy.featureValue]) {const th=document.createElement('th');th.scope='col';th.textContent=label;row.append(th);}
        head.append(row);table.append(head);
        const body=document.createElement('tbody');
        for (const feature of data.features.slice(offset,offset+count)) {
          const tr=document.createElement('tr'),th=document.createElement('th'),td=document.createElement('td');
          th.scope='row';th.textContent=feature.name;
          td.textContent=new Intl.NumberFormat('en-US',{maximumSignificantDigits:6}).format(feature.value);
          tr.append(th,td);body.append(tr);
        }
        offset+=count;table.append(body);article.append(table);groups.append(article);
      }
      root.querySelector('[data-result]').hidden = false;
      status.textContent = copy.success;
    } catch (exception) {
      fail(exception.name === 'AbortError' ? 'timeout' : exception.message);
    } finally {
      clearTimeout(timer);
      submit.disabled = false; input.disabled = false;
      submit.textContent = copy.analyze; form.removeAttribute('aria-busy');
    }
  });
  root.querySelector('[data-download]').addEventListener('click', () => {
    if (!result) return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;link.download='humanorai-result.json';link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
}
