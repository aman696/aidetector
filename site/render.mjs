const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
const arrow = '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>';
const mark = '<svg class="brand-mark" aria-hidden="true" viewBox="0 0 32 32"><path d="M11 3H3v8m18-8h8v8M3 21v8h8m18-8v8h-8"/><path d="M10 16h12m-6-6v12"/></svg>';
const link = (href, label, className = '') => `<a class="${className}" href="${escape(href)}">${escape(label)}${className.includes('button') ? arrow : ''}</a>`;

function heroArt(c) {
  let grid = '';
  for (let i = 0; i < 13; i++) {
    const p = 48 + i * 32;
    grid += `<path d="M48 ${p}H432M${p} 48V432"/>`;
  }
  let waves = '';
  for (let row = 0; row < 27; row++) {
    let path = '';
    for (let column = 0; column <= 75; column++) {
      const x = 80 + column * 4.3;
      const y = 110 + row * 7.5 + Math.sin(column / 9 + row / 4) * 11 + Math.cos(column / 6 - row / 6) * 8;
      path += `${column === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
    }
    waves += `<path d="${path}" opacity="${(0.24 + row / 40).toFixed(2)}"/>`;
  }
  return `<figure class="hero-figure" aria-label="${escape(c.figureLabel)}">
    <div class="figure-top"><span class="dot"></span>${escape(c.figureTop)}<span aria-hidden="true">↗</span></div>
    <svg class="hero-art" viewBox="0 0 480 460" aria-hidden="true">
      <g class="art-grid" fill="none">${grid}</g>
      <g transform="translate(48 47) rotate(-10 225 220)">
        <rect class="art-back" x="64" y="52" width="304" height="284" rx="8"/>
        <rect class="art-middle" x="45" y="70" width="304" height="284" rx="8"/>
        <rect class="art-front" x="26" y="88" width="304" height="284" rx="8"/>
        <g transform="translate(-31 12)" class="art-waves" fill="none">${waves}</g>
        <path class="art-scan" d="M27 211h302"/>
        <circle class="art-node" cx="155" cy="211" r="5"/>
        <circle class="art-node" cx="244" cy="211" r="5"/>
        <g class="art-corners" fill="none"><path d="M37 114v-16h16m251 0h16v16M37 344v16h16m251 0h16v-16"/></g>
      </g>
      <g class="art-callout" fill="none"><path d="M321 100h60v-28h44M331 227h66v16h28M290 388h99v-22h36"/><circle cx="321" cy="100" r="3"/><circle cx="331" cy="227" r="3"/><circle cx="290" cy="388" r="3"/></g>
      <g class="art-ticks"><path d="M40 35h12m-6-6v12M422 411h12m-6-6v12"/></g>
    </svg>
    <div class="figure-bottom"><span>${escape(c.figureBottom)}</span><span>${escape(c.figureDimensions)}</span></div>
    <figcaption>${escape(c.figureCaption)}</figcaption>
  </figure>`;
}

function pipeline(c) {
  const l = c.labels;
  const box = (x,y,w,h,title,detail,accent=false) => `<g class="${accent ? 'pipeline-accent' : 'pipeline-box'}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7"/><text class="pipeline-title" x="${x+w/2}" y="${y+31}" text-anchor="middle">${escape(title)}</text><text class="pipeline-detail" x="${x+w/2}" y="${y+53}" text-anchor="middle">${escape(detail)}</text></g>`;
  const desktop = `<svg class="pipeline pipeline-desktop" role="img" aria-labelledby="pipeline-title pipeline-desc" viewBox="0 0 1080 282">
    <title id="pipeline-title">${escape(c.diagramTitle)}</title><desc id="pipeline-desc">${escape(c.diagramDescription)}</desc>
    <defs><marker id="flow-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5 0 7"/></marker></defs>
    <g class="pipeline-lines" fill="none"><path d="M185 141H218V44H255"/><path d="M218 141h37"/><path d="M218 141v97h37"/><path d="M470 44h40v97h47"/><path d="M470 141h87"/><path d="M470 238h40v-97"/><path d="M757 141h59"/></g>
    ${box(15,101,170,80,l[0],l[1])}${box(255,4,215,80,l[2],l[3])}${box(255,101,215,80,l[4],l[5])}${box(255,198,215,80,l[6],l[7])}${box(557,101,200,80,l[8],l[9])}${box(816,101,249,80,l[10],l[11],true)}
  </svg>`;
  const mobile = `<svg class="pipeline pipeline-mobile" role="img" aria-labelledby="pipeline-mobile-title pipeline-mobile-desc" viewBox="0 0 400 604">
    <title id="pipeline-mobile-title">${escape(c.diagramTitle)}</title><desc id="pipeline-mobile-desc">${escape(c.diagramDescription)}</desc>
    <defs><marker id="mobile-flow-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5 0 7"/></marker></defs>
    <g class="pipeline-mobile-lines" fill="none"><path d="M200 79v15H8v248"/><path d="M8 150h10M8 246h10M8 342h10"/><path d="M382 150h10v288h-52"/><path d="M382 246h10M382 342h10"/><path d="M200 478v40"/></g>
    ${box(60,4,280,75,l[0],l[1])}${box(18,110,364,80,l[2],l[3])}${box(18,206,364,80,l[4],l[5])}${box(18,302,364,80,l[6],l[7])}${box(60,398,280,80,l[8],l[9])}${box(18,518,364,80,l[10],l[11],true)}
  </svg>`;
  return desktop + mobile;
}

function analyzerIcon(type) {
  const shapes = { spectrum: 'M4 25v-7m7 7V9m7 16V4m7 21V13m7 12v-9', texture: 'M4 4h8v8H4zm16 0h8v8h-8zM4 20h8v8H4zm16 0h8v8h-8z', noise: 'M3 16h4l4-10 6 21 5-17 4 6h7' };
  return `<svg class="analyzer-icon" aria-hidden="true" viewBox="0 0 36 36"><path d="${shapes[type]}"/></svg>`;
}

function header(c) {
  return `<a class="skip-link" href="#main">${escape(c.skip)}</a><header class="site-header"><div class="container header-inner">
    <a class="brand" href="/">${mark}<span>${escape(c.brand)}</span></a>
    <nav aria-label="${escape(c.navigationLabel)}">${c.navigation.map(item => link(item.href,item.label)).join('')}</nav>
    <div class="header-actions"><button class="theme-toggle" type="button" aria-label="${escape(c.theme.label)}" aria-pressed="false" data-light-label="${escape(c.theme.label)}" data-dark-label="${escape(c.theme.darkLabel)}"><svg class="theme-moon" aria-hidden="true" viewBox="0 0 24 24"><path d="M20 14a8 8 0 0 1-10-10 8.5 8.5 0 1 0 10 10Z"/></svg><svg class="theme-sun" aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg></button>${link(c.navActionHref,c.navAction,'button button-small button-dark')}</div>
  </div></header>`;
}

function footer(c) {
  return `<footer class="site-footer"><div class="container"><div class="footer-top"><div><a class="brand" href="/">${mark}<span>${escape(c.brand)}</span></a><p>${escape(c.footer.statement)}</p></div><nav aria-label="${escape(c.footerNavigationLabel)}">${c.footer.links.map(item => link(item.href,item.label)).join('')}</nav></div><div class="footer-bottom"><span>${escape(c.footer.copyright)}</span><span>${escape(c.footer.note)}</span></div></div></footer>`;
}

function home(c, demoEnabled) {
  return `<main id="main">
    <section class="hero container" aria-labelledby="hero-title"><div class="hero-copy"><p class="eyebrow">${escape(c.hero.eyebrow)}</p><h1 id="hero-title">${escape(c.hero.title[0])}<span>${escape(c.hero.title[1])}</span></h1><p class="hero-description">${escape(c.hero.description)}</p><div class="hero-actions">${link(c.hero.primaryHref,c.hero.primary,'button button-primary')}${link('#method',c.hero.secondary,'button button-secondary')}</div><p class="hero-note"><span class="note-icon" aria-hidden="true">i</span>${escape(c.hero.note)}</p></div>${heroArt(c.hero)}</section>
    <div class="facts-strip"><div class="container facts-inner">${c.hero.facts.map(fact=>`<div><strong>${escape(fact.value)}</strong><span>${escape(fact.label)}</span></div>`).join('')}</div></div>
    <section id="project" class="section container product-section" aria-labelledby="project-title"><div class="section-heading"><p class="eyebrow">${escape(c.project.eyebrow)}</p><h2 id="project-title">${escape(c.project.title)}</h2><p>${escape(c.project.description)}</p></div><div class="product-grid">${c.project.items.map((item,i)=>`<article><span class="product-number" aria-hidden="true">0${i+1}</span><h3>${escape(item.title)}</h3><p>${escape(item.text)}</p></article>`).join('')}</div></section>
    <section id="method" class="section container" aria-labelledby="method-title"><div class="section-heading"><p class="eyebrow">${escape(c.method.eyebrow)}</p><h2 id="method-title">${escape(c.method.title)}</h2><p>${escape(c.method.description)}</p></div><div class="pipeline-wrap">${pipeline(c.method)}</div><div class="method-steps">${c.method.steps.map(step=>`<article><span class="step-number">${escape(step.number)}</span><h3>${escape(step.title)}</h3><p>${escape(step.text)}</p></article>`).join('')}</div><p class="section-note">${escape(c.method.fallback)}</p></section>
    <section id="analyzers" class="analyzers-section" aria-labelledby="analyzers-title"><div class="section container"><div class="section-heading"><p class="eyebrow">${escape(c.analyzers.eyebrow)}</p><h2 id="analyzers-title">${escape(c.analyzers.title)}</h2><p>${escape(c.analyzers.description)}</p></div><div class="analyzer-grid">${c.analyzers.groups.map(group=>`<article class="analyzer-group">${analyzerIcon(group.icon)}<h3>${escape(group.name)}</h3><p>${escape(group.description)}</p><dl>${group.items.map(([name,count])=>`<div><dt>${escape(name)}</dt><dd>${count}<span class="sr-only"> ${escape(c.analyzers.featureLabel)}</span></dd></div>`).join('')}</dl></article>`).join('')}</div><div class="learned-note"><span class="embedding-mark" aria-hidden="true">${escape(c.analyzers.embeddingDimensions)}</span><div><h3>${escape(c.analyzers.learned.title)}</h3><p>${escape(c.analyzers.learned.text)}</p></div></div><p class="section-note">${escape(c.analyzers.caveat)}</p></div></section>
    <section id="results" class="section container" aria-labelledby="results-title"><div class="section-heading"><p class="eyebrow">${escape(c.results.eyebrow)}</p><h2 id="results-title">${escape(c.results.title)}</h2><p>${escape(c.results.description)}</p></div><div class="evidence-grid"><article class="evidence-panel measured-panel"><p class="panel-kicker">${escape(c.results.measuredTitle)}</p><p class="panel-subtitle">${escape(c.results.measuredSubtitle)}</p><dl class="headline-metrics">${c.results.metrics.map(metric=>`<div><dt>${escape(metric.label)}<span>${escape(metric.context)}</span></dt><dd>${escape(metric.value)}</dd></div>`).join('')}</dl></article><article class="evidence-panel limits-panel"><p class="panel-kicker">${escape(c.results.limitsTitle)}</p><p class="panel-subtitle">${escape(c.results.limitsSubtitle)}</p><ul>${c.results.limits.map(limit=>`<li><h3>${escape(limit.title)}</h3><p>${escape(limit.text)}</p></li>`).join('')}</ul></article></div>
    <div class="benchmark-context"><h3>${escape(c.results.contextTitle)}</h3><div>${c.results.context.map(text=>`<p>${escape(text)}</p>`).join('')}</div></div>
    <div class="table-wrap"><table class="benchmark-table"><caption>${escape(c.results.tableCaption)}</caption><thead><tr>${c.results.tableHeaders.map(label=>`<th scope="col">${escape(label)}</th>`).join('')}</tr></thead><tbody>${c.results.tableRows.map(row=>`<tr>${row.map((value,i)=>i===0?`<th scope="row">${escape(value)}</th>`:`<td>${escape(value)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
    <div class="gates-panel"><h3>${escape(c.results.gatesTitle)}</h3><div class="gates-grid">${c.results.gates.map(gate=>`<div><h4>${escape(gate.name)}</h4><p>${escape(c.results.measuredLabel)} <strong>${escape(gate.value)}</strong><span>${escape(c.results.targetLabel)} ${escape(gate.target)}</span></p></div>`).join('')}</div><p>${escape(c.results.gatesNote)}</p></div>
    <div class="condition-explorer"><div><h3>${escape(c.results.conditionsTitle)}</h3><p>${escape(c.results.conditionsDescription)}</p></div><div class="condition-controls" role="group" aria-label="${escape(c.results.conditionsAria)}">${c.results.conditions.map((condition,i)=>`<button type="button" data-condition="${i}" aria-pressed="${i===0}">${escape(condition.label)}</button>`).join('')}</div><div class="condition-panels" aria-live="polite" aria-atomic="true">${c.results.conditions.map((condition,i)=>`<div class="condition-stats" data-condition-panel="${i}" ${i===0?'':'hidden'}>${[condition.rows,condition.accuracy,condition.auc,condition.detection].map((value,j)=>`<div><span>${escape(c.results.conditionLabels[j])}</span><strong>${escape(value)}</strong></div>`).join('')}</div>`).join('')}</div></div>
    <div class="evidence-links">${link('/data/evaluation.json',c.results.evidenceLink,'text-link')}${link('/docs/MODEL_CARD.md',c.results.modelCardLink,'text-link')}</div></section>
    <section id="direction" class="direction-section" aria-labelledby="direction-title"><div class="section container"><div class="section-heading"><p class="eyebrow">${escape(c.direction.eyebrow)}</p><h2 id="direction-title">${escape(c.direction.title)}</h2><p>${escape(c.direction.description)}</p></div><div class="direction-grid">${c.direction.items.map(item=>`<article><p class="eyebrow">${escape(item.stage)}</p><h3>${escape(item.title)}</h3><p>${escape(item.text)}</p></article>`).join('')}</div><p class="direction-note">${escape(c.direction.note)}</p></div></section>
    <section id="get-started" class="start-section" aria-labelledby="start-title"><div class="section container start-grid"><div class="start-copy"><p class="eyebrow">${escape(c.start.eyebrow)}</p><h2 id="start-title">${escape(c.start.title)}</h2><p>${escape(c.start.description)}</p><p class="prerequisite">${escape(c.start.prerequisite)}</p><div class="start-links">${link(c.start.githubUrl,c.start.github,'button button-primary')}${link(c.start.datasetUrl,c.start.dataset,'text-link')}</div><ul>${c.start.notes.map(note=>`<li>${escape(note)}</li>`).join('')}</ul></div><div class="code-panel"><div class="code-header"><span>${escape(c.start.commandsTitle)}</span><button class="copy-button" data-copy type="button" data-copied="${escape(c.start.copied)}" data-failed="${escape(c.start.copyFailed)}">${escape(c.start.copy)}</button></div><pre tabindex="0"><code id="start-commands">${escape(c.start.commands)}</code></pre><span class="sr-only" data-copy-status role="status"></span></div></div></section>
    <section id="demo" class="section container demo-section" aria-labelledby="demo-title"><p class="eyebrow">${escape(c.demo.eyebrow)}</p><h2 id="demo-title">${escape(c.demo.title)}</h2><div class="demo-panel"><div class="demo-icon">${mark}</div><div><h3>${escape(demoEnabled?c.demo.title:c.demo.offline)}</h3><p>${escape(demoEnabled?c.demo.enabledDescription:c.demo.description)}</p></div>${link(demoEnabled?'/test/':'#get-started',demoEnabled?c.demo.enabledButton:c.demo.button,'button button-secondary')}</div></section>
  </main>`;
}

function resources(c) {
  return `<main id="main" class="container resources-main"><p class="eyebrow">${escape(c.resources.eyebrow)}</p><h1>${escape(c.resources.title)}</h1><p class="resource-intro">${escape(c.resources.description)}</p><div class="resource-list">${c.resources.entries.map(entry=>`<a class="resource-row" href="${escape(entry.href)}"><div><h2>${escape(entry.label)}</h2><p>${escape(entry.description)}</p></div><span class="resource-type">${escape(entry.type)}</span>${arrow}</a>`).join('')}</div>${link('/',c.resources.back,'button button-secondary')}</main>`;
}

function testPage(c, enabled, apiOrigin) {
  const t = c.test;
  const interactive = enabled ? `<section class="upload-panel" data-demo data-api-origin="${escape(apiOrigin)}" data-demo-copy="${escape(JSON.stringify(t))}" aria-label="${escape(t.inputLabel)}">
    <form id="demo-form"><label for="demo-file">${escape(t.inputLabel)}</label><p id="file-hint">${escape(t.inputHint)}</p><input id="demo-file" type="file" accept="image/jpeg,image/png,image/webp" aria-describedby="file-hint" required><div class="selected-file" hidden><img data-preview alt="${escape(t.previewAlt)}"><p><span>${escape(t.fileLabel)}</span><strong data-filename></strong></p></div><p data-preparation hidden></p><button class="button button-primary" type="submit">${escape(t.analyze)}</button></form>
    <p class="demo-status" role="status" aria-live="polite" data-demo-status>${escape(t.ready)}</p><p class="demo-error" role="alert" data-demo-error hidden></p>
    <section class="demo-result" data-result hidden aria-labelledby="estimate-title"><p class="eyebrow">${escape(t.modelLabel)}</p><h2 id="estimate-title">${escape(t.probabilityLabel)}</h2><strong class="demo-probability" data-probability></strong><progress data-probability-bar max="1" value="0" aria-label="${escape(t.probabilityLabel)}"></progress><dl class="demo-result-meta"><div><dt>${escape(t.verdictLabel)}</dt><dd data-verdict></dd></div><div><dt>${escape(t.durationLabel)}</dt><dd data-duration></dd></div></dl><p data-processing></p><p>${escape(t.resultNote)}</p><button type="button" class="button button-secondary" data-download>${escape(t.download)}</button><details class="demo-features"><summary>${escape(t.featuresTitle)}</summary><p>${escape(t.featuresHint)}</p><div data-feature-groups></div></details></section>
  </section>` : `<div class="demo-panel"><div><h2>${escape(c.demo.offline)}</h2><p>${escape(c.demo.description)}</p></div>${link('/#get-started',c.demo.button,'button button-secondary')}</div>`;
  return `<main id="main" class="container test-main"><p class="eyebrow">${escape(t.eyebrow)}</p><h1>${escape(t.title)}</h1><p class="test-intro">${escape(t.description)}</p><div class="test-grid"><div>${interactive}</div><aside class="demo-context"><h2>${escape(t.modelTitle)}</h2><p>${escape(t.modelDescription)}</p><p>${escape(t.benchmark)}</p><p>${escape(t.limits)}</p>${enabled?`<h2>${escape(t.privacyTitle)}</h2><p>${escape(t.privacy)}</p>${link(t.privacyHref,t.privacyLink,'text-link')}<p class="demo-cold-start">${escape(t.coldStart)}</p>`:''}${link('/#results',c.results.title,'text-link')}</aside></div></main>`;
}

export function renderPage(c, options) {
  const { page, siteUrl, assets, demoEnabled } = options;
  const path = page==='test'?'/test/':page==='resources'?'/resources/':page==='404'?'/404.html':'/';
  const title = page==='test'?`${c.test.title} — ${c.brand}`:page==='resources'?`${c.resources.title} — ${c.brand}`:page==='404'?`${c.notFound.title} — ${c.brand}`:c.title;
  const schema = JSON.stringify({ '@context':'https://schema.org','@type':'SoftwareApplication',name:c.brand,description:c.description,url:siteUrl,applicationCategory:'EducationalApplication',operatingSystem:'Python 3.10+',isAccessibleForFree:true,codeRepository:c.start.githubUrl });
  let main = page==='test'?testPage(c,demoEnabled,options.demoApiOrigin):page==='resources'?resources(c):page==='404'?`<main id="main" class="container not-found"><h1>${escape(c.notFound.title)}</h1><p>${escape(c.notFound.description)}</p>${link('/',c.notFound.action,'button button-primary')}</main>`:home(c,demoEnabled);
  return { schema, html:`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escape(title)}</title><meta name="description" content="${escape(c.description)}"><meta name="color-scheme" content="light dark"><meta name="theme-color" content="#f8f9fc">${page==='404'?'<meta name="robots" content="noindex">':`<link rel="canonical" href="${escape(siteUrl+path)}">`}<meta property="og:type" content="website"><meta property="og:site_name" content="${escape(c.brand)}"><meta property="og:title" content="${escape(title)}"><meta property="og:description" content="${escape(c.description)}"><meta property="og:url" content="${escape(siteUrl+path)}"><meta property="og:image" content="${escape(siteUrl)}/opengraph.png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta property="og:image:alt" content="${escape(c.socialAlt)}"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="${escape(title)}"><meta name="twitter:description" content="${escape(c.description)}"><meta name="twitter:image" content="${escape(siteUrl)}/opengraph.png"><meta name="twitter:image:alt" content="${escape(c.socialAlt)}"><link rel="icon" href="/favicon.svg" type="image/svg+xml"><script src="${assets.theme}"></script><link rel="stylesheet" href="${assets.css}"><script defer src="${assets.js}"></script>${page==='test'&&demoEnabled?`<script defer src="${assets.demo}"></script>`:''}<script type="application/ld+json">${schema}</script></head><body>${header(c)}${main}${footer(c)}</body></html>` };
}
