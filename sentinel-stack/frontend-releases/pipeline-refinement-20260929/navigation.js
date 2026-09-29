// Honor deep links after React mounts; do not move the page once the user scrolls.
(() => {
  const pending = location.hash.slice(1);
  let interrupted = false;
  let resize;
  const stop = () => { interrupted = true; observer.disconnect(); resize?.disconnect(); };
  const resolve = () => {
    if (!pending || interrupted) return;
    const target = document.getElementById(pending);
    if (!target) return;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (!interrupted) target.scrollIntoView({ block: 'start', behavior: 'instant' });
    }));
    if (!resize) {
      resize = new ResizeObserver(resolve);
      resize.observe(document.querySelector('main'));
    }
    observer.disconnect();
  };
  const observer = new MutationObserver(resolve);
  observer.observe(document.getElementById('root'), { childList: true, subtree: true });
  addEventListener('wheel', stop, { once: true, passive: true });
  addEventListener('touchstart', stop, { once: true, passive: true });
  addEventListener('pointerdown', stop, { once: true, passive: true });
  addEventListener('keydown', stop, { once: true });
  addEventListener('hashchange', stop, { once: true });
  setTimeout(stop, 5000);
  resolve();
})();
