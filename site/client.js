const themeButton = document.querySelector('.theme-toggle');
const updateThemeButton = () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  themeButton.setAttribute('aria-pressed', String(dark));
  themeButton.setAttribute('aria-label', dark ? themeButton.dataset.darkLabel : themeButton.dataset.lightLabel);
};
updateThemeButton();
themeButton.addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('human-or-ai-theme', theme); } catch {}
  updateThemeButton();
});

for (const button of document.querySelectorAll('[data-condition]')) {
  button.addEventListener('click', () => {
    for (const control of document.querySelectorAll('[data-condition]')) {
      control.setAttribute('aria-pressed', String(control === button));
    }
    for (const panel of document.querySelectorAll('[data-condition-panel]')) {
      panel.hidden = panel.dataset.conditionPanel !== button.dataset.condition;
    }
  });
}

const copyButton = document.querySelector('[data-copy]');
copyButton?.addEventListener('click', async () => {
  const status = document.querySelector('[data-copy-status]');
  try {
    await navigator.clipboard.writeText(document.querySelector('#start-commands').textContent);
    status.textContent = copyButton.dataset.copied;
    copyButton.textContent = copyButton.dataset.copied;
  } catch {
    status.textContent = copyButton.dataset.failed;
    document.querySelector('.code-panel pre').focus();
  }
});
