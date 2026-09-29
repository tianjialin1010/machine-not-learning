"""One-time, hash-guarded PC UI release for the inspected Spark service."""

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

ROOT = Path('/home/USER/sentinel-stack')
STAGE = Path(__file__).resolve().parent
EXPECTED_SOURCE = 'bae6f63073d2c20ec29eba0ff3127b7eff0c4a3169d7f4ebb129d0124f61ee77'
EXPECTED_PATCH = '5329783224abfaf3067d800811baaf74f6b4a747b242c5cd23117ebcf6b864ee'
PID = 2877988
SOURCE = ROOT / 'sentinel_integrated.py'
HELPER = ROOT / 'frontend_static.py'
CURRENT = ROOT / 'frontend-current'
BACKUP = ROOT / 'deployment-backups' / STAGE.name
RELEASE = ROOT / 'frontend-releases' / STAGE.name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get(path):
    with urlopen('http://127.0.0.1:9000' + path, timeout=3) as response:
        return response.read()


def atomic_copy(source, destination):
    temporary = destination.with_name(destination.name + '.deploy-next')
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def stop(pid):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(100):
        try:
            state = Path(f'/proc/{pid}/stat').read_text().split()[2]
        except FileNotFoundError:
            return
        if state == 'Z':
            return
        time.sleep(.1)
    raise RuntimeError(f'Process {pid} did not stop; no forced kill performed')


def start():
    with (ROOT / 'service-9000.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    return process


def wait_ready(process, frontend):
    for _ in range(45):
        if process.poll() is not None:
            raise RuntimeError('Service exited during startup; inspect service-9000.log')
        try:
            health = json.loads(get('/healthz'))
            for key in ('ok', 'device', 'decision_head', 'feature_contract', 'policy', 'generative_tier', 'temperatures'):
                if health.get(key) != before_health.get(key):
                    raise ValueError('Health metadata mismatch: ' + key)
            if frontend:
                config = json.loads(get('/api/config'))
                assert config['decision_path'] == '/v1/decide'
                assert config['upstream_health']['ok'] is True
                assert get('/') == (RELEASE / 'index.html').read_bytes()
                assert get('/models/ai4i.glb') == (RELEASE / 'models/ai4i.glb').read_bytes()
                assert get('/dashboard') == (ROOT / 'dashboard.html').read_bytes()
            return health
        except Exception:
            time.sleep(1)
    raise RuntimeError('Service verification timed out')


if __name__ == '__main__':
    if STAGE.parent != ROOT / 'deploy-staging':
        sys.exit('Run only from the inspected deploy-staging release directory')
    manifest = json.loads((STAGE / 'manifest.json').read_text())
    for relative, digest in manifest.items():
        assert sha(STAGE / relative) == digest, 'Staged file differs: ' + relative
    assert sha(SOURCE) == EXPECTED_SOURCE, 'Backend source changed; stop for review'
    assert sha(STAGE / 'sentinel_integrated.py') == EXPECTED_PATCH
    assert not HELPER.exists() and not HELPER.is_symlink() and not CURRENT.exists() and not CURRENT.is_symlink(), 'Unexpected prior frontend installation'
    assert not BACKUP.exists() and not RELEASE.exists(), 'Release name already exists'
    command = [part.decode() for part in Path(f'/proc/{PID}/cmdline').read_bytes().split(b'\0') if part]
    assert command == ['python3', '-u', 'sentinel_integrated.py', '--port', '9000'], 'Running command changed'
    assert Path(f'/proc/{PID}/cwd').resolve() == ROOT, 'Running directory changed'
    # Preserve the existing process environment in memory, without logging it.
    environment = dict(part.decode().split('=', 1) for part in Path(f'/proc/{PID}/environ').read_bytes().split(b'\0') if part)
    before_health = json.loads(get('/healthz'))
    protected = {name: sha(ROOT / name) for name in ('sentinel_heads.py', 'heads.pt', 'heads_meta.json', 'dashboard.html', 'data/ai4i2020.csv')}
    for name in ('sentinel_integrated.py', 'frontend_static.py'):
        compile((STAGE / name).read_bytes(), name, 'exec')
    BACKUP.mkdir(parents=True)
    shutil.copy2(SOURCE, BACKUP / SOURCE.name)
    (BACKUP / 'health-before.json').write_text(json.dumps(before_health, indent=2))
    (BACKUP / 'protected-sha256.json').write_text(json.dumps(protected, indent=2))
    RELEASE.parent.mkdir(exist_ok=True)
    shutil.copytree(STAGE / 'frontend', RELEASE)
    # Recheck immediately before the first live change.
    assert sha(SOURCE) == EXPECTED_SOURCE, 'Backend changed during staging'
    stop(PID)
    process = None
    try:
        atomic_copy(STAGE / 'frontend_static.py', HELPER)
        atomic_copy(STAGE / 'sentinel_integrated.py', SOURCE)
        CURRENT.symlink_to(RELEASE, target_is_directory=True)
        process = start()
        health = wait_ready(process, frontend=True)
        assert protected == {name: sha(ROOT / name) for name in protected}, 'Protected backend artifacts changed'
        result = {'status': 'deployed', 'release': str(RELEASE), 'backup': str(BACKUP), 'pid': process.pid, 'decision_head': health['decision_head'], 'protected_artifacts_unchanged': True}
        (BACKUP / 'deployment-result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except Exception:
        if process is not None and process.poll() is None:
            stop(process.pid)
        atomic_copy(BACKUP / SOURCE.name, SOURCE)
        if CURRENT.is_symlink() and CURRENT.resolve() == RELEASE:
            CURRENT.unlink()
        if HELPER.exists():
            HELPER.unlink()
        restored = start()
        wait_ready(restored, frontend=False)
        print('Deployment verification failed; original service restored', file=sys.stderr, flush=True)
        raise
