"""Install this exact frontend release, with compare-before-swap and rollback."""
from pathlib import Path
import hashlib
import json
import os
import tarfile
from urllib.request import urlopen

root = Path('/home/USER/sentinel-stack')
link = root / 'frontend-current'
previous = root / 'frontend-releases/scenario-layout-20260929'
release = root / 'frontend-releases/pipeline-refinement-20260929'
archive = root / 'frontend-releases/pipeline-refinement-20260929.tar.gz'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert link.resolve() == previous, 'Live release changed; stop without replacing it.'
assert sha(previous / 'assets/index-BoH9Y5xq.js') == '411da604a9c85b12c1bb927d41d7f4ed76fac00b88231506f9a687b686c2cdf8', 'Live bundle changed.'
assert not release.exists(), 'Release already exists; do not overwrite.'
protected = ['sentinel_integrated.py', 'industrial_runtime.py', 'sentinel_heads.py', 'heads.pt', 'frontend_static.py']
before = {name: sha(root / name) for name in protected}
release.mkdir()
with tarfile.open(archive) as tar:
    tar.extractall(release, filter='data')
manifest = json.loads((release / 'release-manifest.json').read_text())
for name, digest in manifest.items():
    assert sha(release / name) == digest, name
assert '看清一次分诊的完整路径' in (release / 'assets/index-BoH9Y5xq.js').read_text()
temporary = root / 'frontend-next-pipeline-refinement'
assert not temporary.exists() and not temporary.is_symlink()
temporary.symlink_to(release)
assert link.resolve() == previous
os.replace(temporary, link)
try:
    for name in ['index.html', 'refinement.css', 'navigation.js', 'assets/index-BoH9Y5xq.js', 'models/ai4i.glb']:
        with urlopen('http://127.0.0.1:9000/' + name, timeout=15) as response:
            assert response.status == 200, name
            assert hashlib.sha256(response.read()).hexdigest() == manifest[name], name
    assert {name: sha(root / name) for name in protected} == before
except Exception:
    temporary.symlink_to(previous)
    os.replace(temporary, link)
    raise
record = {'status': 'published', 'previous': str(previous), 'current': str(release), 'backend_restarted': False, 'protected_hashes_unchanged': before, 'verified_files': len(manifest)}
(root / 'frontend-releases/pipeline-refinement-20260929-result.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
