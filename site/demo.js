const root = document.querySelector('[data-demo]');
if (root) {
  const copy = JSON.parse(root.dataset.demoCopy);
  const origin = root.dataset.apiOrigin;
  const form = root.querySelector('form');
  const input = root.querySelector('input');
  const submit = form.querySelector('button');
  const status = root.querySelector('[data-demo-status]');
  const error = root.querySelector('[data-demo-error]');
  let previewUrl;
  let result;
  let fileIsValid = false;
  const fail = code => {
    error.textContent = copy.errors[code] || copy.errors.offline;
    error.hidden = false;
    status.textContent = '';
  };
  input.addEventListener('change', () => {
    fileIsValid = false;
    error.hidden = true;
    root.querySelector('[data-result]').hidden = true;
    root.querySelector('.selected-file').hidden = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const file = input.files[0];
    if (!file) return;
    if (!/\.(jpe?g|png|webp)$/i.test(file.name)) return fail('invalid_type');
    if (file.size > 1048576) return fail('too_large');
    previewUrl = URL.createObjectURL(file);
    const image = root.querySelector('[data-preview]');
    image.onload = () => {
      if (image.naturalWidth * image.naturalHeight > 1048576 || Math.max(image.naturalWidth,image.naturalHeight) > 4096) return fail('dimensions');
      fileIsValid = true;
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
      status.textContent = copy.checking;
      const health = await fetch(origin+'/healthz',{signal:controller.signal,cache:'no-store',credentials:'omit'});
      const ready = await health.json();
      if (!health.ok || !ready.model_loaded || ready.method !== 'classical_v2') throw new Error('offline');
      status.textContent = copy.analyzing;
      const body = new FormData(); body.append('file',input.files[0]);
      const response = await fetch(origin+'/api/detect',{method:'POST',body,signal:controller.signal,credentials:'omit'});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail==='string'?data.detail:response.status===429?'rate_limit':'analysis_failed');
      if (!validResult(data)) throw new Error('invalid_result');
      result = data;
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
