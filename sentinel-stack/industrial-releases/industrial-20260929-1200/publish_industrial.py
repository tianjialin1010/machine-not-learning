"""Publish the independently validated five-scenario candidate, with rollback."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path('/home/USER/sentinel-stack')
RELEASE = Path(__file__).resolve().parent
CANDIDATE = RELEASE / 'candidate'
BACKUP = ROOT / 'deployment-backups' / RELEASE.name
EXPECTED = '5329783224abfaf3067d800811baaf74f6b4a747b242c5cd23117ebcf6b864ee'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get(path, port=9000):
    with urlopen(f'http://127.0.0.1:{port}{path}', timeout=10) as response:
        return response.read()


def copy_atomic(source, target):
    temporary = target.with_name(target.name + '.next-' + uuid4().hex)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def link_atomic(target, link):
    temporary = link.with_name(link.name + '.next-' + uuid4().hex)
    try:
        temporary.symlink_to(target, target_is_directory=True)
        temporary.replace(link)
    finally:
        temporary.unlink(missing_ok=True)


def stop(pid):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(150):
        try:
            state = Path(f'/proc/{pid}/stat').read_text().split()[2]
        except FileNotFoundError:
            return
        if state == 'Z':
            return
        time.sleep(.1)
    raise RuntimeError('Process did not stop; no files replaced')


def start(command, environment):
    with (ROOT / 'service-9000.log').open('ab') as log:
        return subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def verify(process, industrial, health_before):
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError('Service exited during startup')
        try:
            health = json.loads(get('/healthz'))
            for key in ('ok', 'device', 'decision_head', 'feature_contract', 'policy', 'temperatures', 'generative_tier'):
                assert health[key] == health_before[key], 'CNC health changed: ' + key
            if industrial:
                catalog = json.loads(get('/v1/industrial/catalog'))
                assert {item['id'] for item in catalog['scenarios'] if item['available']} == {'ai4i', 'skab', 'steel', 'secom', 'tep'}
                assert get('/') == (RELEASE / 'frontend/index.html').read_bytes()
            return health
        except Exception:
            time.sleep(1)
    raise RuntimeError('Startup verification timed out')


if __name__ == '__main__':
    assert RELEASE.parent == ROOT / 'industrial-releases'
    assert not BACKUP.exists(), 'Release already attempted; review existing backup'
    assert sha(ROOT / 'sentinel_integrated.py') == EXPECTED, 'Backend changed; stop for review'
    assert not os.path.lexists(ROOT / 'industrial_runtime.py'), 'Unexpected existing runtime'
    assert not os.path.lexists(ROOT / 'industrial-models-current'), 'Unexpected existing model link'
    assert (ROOT / 'frontend-current').is_symlink()
    assert (RELEASE / 'candidate-smoke.json').is_file(), 'Candidate must pass real inference before publishing'
    smoke = json.loads((RELEASE / 'candidate-smoke.json').read_text())
    assert smoke['ok'] and set(smoke['scenarios']) == {'ai4i', 'skab', 'steel', 'secom', 'tep'}
    manifest = json.loads((RELEASE / 'publish-manifest.json').read_text())
    for relative, digest in manifest.items():
        assert sha(RELEASE / relative) == digest, 'Release file differs: ' + relative
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
            if command == ['python3', '-u', 'sentinel_integrated.py', '--port', '9000'] and (proc / 'cwd').resolve() == ROOT:
                matches.append((int(proc.name), command))
        except (OSError, UnicodeError):
            continue
    assert len(matches) == 1, 'Expected one inspected 9000 process'
    pid, command = matches[0]
    environment = dict(part.decode().split('=', 1) for part in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if part)
    health_before = json.loads(get('/healthz'))
    protected = {name: sha(ROOT / name) for name in ('sentinel_heads.py', 'heads.pt', 'heads_meta.json', 'dashboard.html', 'data/ai4i2020.csv')}
    original_frontend = (ROOT / 'frontend-current').resolve()
    for name in ('sentinel_integrated.py', 'industrial_runtime.py'):
        compile((CANDIDATE / name).read_bytes(), name, 'exec')
    BACKUP.mkdir(parents=True)
    shutil.copy2(ROOT / 'sentinel_integrated.py', BACKUP / 'sentinel_integrated.py')
    (BACKUP / 'before.json').write_text(json.dumps({'health': health_before, 'protected': protected, 'frontend': str(original_frontend), 'command': command}, indent=2))
    assert sha(ROOT / 'sentinel_integrated.py') == EXPECTED
    stop(pid)
    process = None
    try:
        copy_atomic(CANDIDATE / 'industrial_runtime.py', ROOT / 'industrial_runtime.py')
        copy_atomic(CANDIDATE / 'sentinel_integrated.py', ROOT / 'sentinel_integrated.py')
        link_atomic(RELEASE / 'artifacts', ROOT / 'industrial-models-current')
        link_atomic(RELEASE / 'frontend', ROOT / 'frontend-current')
        process = start(command, environment)
        verify(process, True, health_before)
        assert protected == {name: sha(ROOT / name) for name in protected}, 'Protected CNC artifacts changed'
        result = {'ok': True, 'release': str(RELEASE), 'backup': str(BACKUP), 'pid': process.pid, 'protected_artifacts_unchanged': True}
        (BACKUP / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except Exception:
        if process is not None and process.poll() is None:
            stop(process.pid)
        copy_atomic(BACKUP / 'sentinel_integrated.py', ROOT / 'sentinel_integrated.py')
        link_atomic(original_frontend, ROOT / 'frontend-current')
        models = ROOT / 'industrial-models-current'
        if models.is_symlink() and models.resolve() == RELEASE / 'artifacts':
            models.unlink()
        runtime = ROOT / 'industrial_runtime.py'
        if runtime.exists() and sha(runtime) == sha(CANDIDATE / runtime.name):
            runtime.unlink()
        restored = start(command, environment)
        verify(restored, False, health_before)
        print('Deployment failed; original service restored', file=sys.stderr, flush=True)
        raise
