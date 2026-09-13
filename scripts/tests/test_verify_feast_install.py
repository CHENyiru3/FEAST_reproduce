"""Regression checks for recorded-build integrity and checkout independence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_feast_install', SCRIPTS / 'verify_feast_install.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def recorded_install(tmp_path, monkeypatch):
    repo = tmp_path / 'FEAST'
    source = repo / 'src/FEAST/__init__.py'
    source.parent.mkdir(parents=True)
    source.write_text('__version__ = "1.0.5"\n')
    (repo / 'pyproject.toml').write_text('[project]\nname = "FEAST-py"\nversion = "1.0.5"\n')

    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

    git('init', '-q')
    git('config', 'user.name', 'Verifier test')
    git('config', 'user.email', 'test@example.invalid')
    git('add', '.')
    git('commit', '-qm', 'recorded package')
    commit = git('rev-parse', 'HEAD')
    installed = tmp_path / 'environment/site-packages/FEAST/__init__.py'
    installed.parent.mkdir(parents=True)
    installed.write_bytes(source.read_bytes())
    build = tmp_path / 'build'
    build.mkdir()
    wheel = build / 'feast_py-1.0.5-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.write(source, 'FEAST/__init__.py')
    record = build / 'provenance.json'
    record.write_text(json.dumps({'base_commit': commit, 'candidate_version': '1.0.5',
                                 'candidate_status': 'test', 'wheel': {'path': wheel.name, 'sha256': verifier.sha256(wheel)}}))
    monkeypatch.setattr(verifier, 'ROOT', tmp_path / 'FEAST_reproduce')
    monkeypatch.setattr(sys, 'argv', ['verify_feast_install.py', '--candidate-provenance', str(record)])
    monkeypatch.setitem(sys.modules, 'FEAST', SimpleNamespace(__version__='1.0.5', __file__=str(installed)))
    monkeypatch.setattr(verifier.importlib.metadata, 'distribution', lambda name: SimpleNamespace(version='1.0.5'))
    return SimpleNamespace(repo=repo, source=source, installed=installed, wheel=wheel, git=git, commit=commit)


def test_documentation_only_head_keeps_recorded_build_identity(recorded_install, capsys):
    build = recorded_install
    (build.repo / 'README.md').write_text('Updated documentation.\n')
    build.git('add', 'README.md')
    build.git('commit', '-qm', 'documentation only')
    assert verifier.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report['commit'] == build.commit
    assert report['checkout_head'] == build.git('rev-parse', 'HEAD')
    assert report['checkout_head'] != report['commit']
    assert report['checkout_package_matches_build'] is True
    assert report['verified_package_files'] == 1


@pytest.mark.parametrize('commit_change', [False, True])
def test_changed_checkout_package_is_rejected(recorded_install, commit_change):
    build = recorded_install
    build.source.write_text('changed implementation\n')
    if commit_change:
        build.git('add', '.')
        build.git('commit', '-qm', 'source change')
    with pytest.raises(RuntimeError, match='checkout package/build files differ'):
        verifier.main()


def test_untracked_package_code_is_rejected(recorded_install):
    (recorded_install.source.parent / 'extra.py').write_text('new implementation\n')
    with pytest.raises(RuntimeError, match='checkout package/build files differ'):
        verifier.main()


def test_modified_installed_package_is_rejected(recorded_install):
    recorded_install.installed.write_text('changed installed implementation\n')
    with pytest.raises(RuntimeError, match='installed wheel files differ'):
        verifier.main()


def test_modified_recorded_wheel_is_rejected(recorded_install):
    with recorded_install.wheel.open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(RuntimeError, match='wheel checksum differs'):
        verifier.main()


def test_source_checkout_import_is_rejected(recorded_install, monkeypatch):
    monkeypatch.setitem(sys.modules, 'FEAST', SimpleNamespace(__version__='1.0.5', __file__=str(recorded_install.source)))
    with pytest.raises(RuntimeError, match='mutable source checkout'):
        verifier.main()


@pytest.mark.parametrize('change', ['legacy', 'stack_size', 'old_calibration', 'calibration_seed', 'transport'])
def test_active_study06_checker_rejects_wrong_design(monkeypatch, change):
    spec = importlib.util.spec_from_file_location('conditional_check', SCRIPTS / 'check_conditional_workflows.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert checker.main() == 0
    read_yaml = checker.read_yaml

    def altered(relative):
        config = read_yaml(relative)
        if relative == '06_3d_stack/config_reference_density.yaml':
            if change == 'legacy':
                config = read_yaml('06_3d_stack/config.yaml')
            elif change == 'stack_size':
                config['validation']['expected_positions_per_stack'] = 147
            elif change == 'old_calibration':
                config['densities'][2]['expected_assignment_randomness'] = 0.35
            elif change == 'calibration_seed':
                config['assignment_randomness_preflight']['seed_formula'] = 'public_seed'
            else:
                config['transport']['transport_nonconvergence'] = 'warn'
        return config

    monkeypatch.setattr(checker, 'read_yaml', altered)
    assert checker.main() == 1


@pytest.fixture
def local_recorded_install(recorded_install):
    build = recorded_install
    build_dir = build.wheel.parent
    (build_dir / 'source.patch').write_text(
        'diff --git a/src/FEAST/__init__.py b/src/FEAST/__init__.py\n'
        '--- a/src/FEAST/__init__.py\n+++ b/src/FEAST/__init__.py\n'
        '@@ -1 +1,2 @@\n __version__ = "1.0.5"\n+LOCAL_BUILD = True\n')
    added = b'LOCAL_MODULE = True\n'
    (build_dir / 'new_local.py').write_bytes(added)
    source = build.source.read_bytes() + b'LOCAL_BUILD = True\n'
    build.source.write_bytes(source)
    build.installed.write_bytes(source)
    for parent in [build.source.parent, build.installed.parent]:
        (parent / 'de_novo').mkdir()
        (parent / 'de_novo/local.py').write_bytes(added)
    with zipfile.ZipFile(build.wheel, 'w') as archive:
        archive.writestr('FEAST/__init__.py', source)
        archive.writestr('FEAST/de_novo/local.py', added)
    record_path = build_dir / 'provenance.json'
    record = json.loads(record_path.read_text())
    record['source_mode'] = 'local_working_tree'
    record['wheel']['sha256'] = verifier.sha256(build.wheel)
    record_path.write_text(json.dumps(record))
    return build


@pytest.mark.parametrize('commit_change', [False, True])
def test_local_recorded_source_survives_checkout_development(local_recorded_install, commit_change, capsys):
    build = local_recorded_install
    assert verifier.main() == 0
    assert json.loads(capsys.readouterr().out)['checkout_package_matches_build'] is True
    build.source.write_text('subsequent implementation\n')
    if commit_change:
        # This is a disposable test repository, never the FEAST checkout.
        build.git('add', '.')
        build.git('commit', '-qm', 'subsequent source change')
    assert verifier.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report['commit'] == build.commit
    assert report['checkout_head'] == build.git('rev-parse', 'HEAD')
    assert report['checkout_package_matches_build'] is False
    assert report['recorded_source_matches_build'] is True
    assert report['verified_package_files'] == 2


@pytest.mark.parametrize('corruption', ['installed', 'wheel', 'patch', 'new_module', 'missing_snapshot', 'version', 'base_commit'])
def test_local_build_integrity_remains_strict(local_recorded_install, monkeypatch, corruption):
    build = local_recorded_install
    build_dir = build.wheel.parent
    expected = RuntimeError
    if corruption == 'installed':
        build.installed.write_text('tampered runtime\n')
    elif corruption == 'wheel':
        with build.wheel.open('ab') as handle:
            handle.write(b'tampered wheel')
    elif corruption == 'patch':
        p = build_dir / 'source.patch'
        p.write_text(p.read_text().replace('LOCAL_BUILD = True', 'LOCAL_BUILD = False'))
    elif corruption == 'new_module':
        (build_dir / 'new_local.py').write_text('tampered source\n')
    elif corruption == 'missing_snapshot':
        (build_dir / 'source.patch').unlink()
    elif corruption == 'version':
        monkeypatch.setitem(sys.modules, 'FEAST', SimpleNamespace(__version__='9.9', __file__=str(build.installed)))
    else:
        p = build_dir / 'provenance.json'
        record = json.loads(p.read_text())
        record['base_commit'] = '0' * 40
        p.write_text(json.dumps(record))
        expected = subprocess.CalledProcessError
    with pytest.raises(expected):
        verifier.main()


def test_study07_identity_uses_verified_build_not_checkout_head(tmp_path, monkeypatch):
    import importlib
    module_spec = importlib.util.spec_from_file_location('study07_identity_test', SCRIPTS.parent / '07_3d_transfer/workflow.py')
    workflow = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(workflow)
    installed = tmp_path / 'environment/site-packages/FEAST/__init__.py'
    monkeypatch.setitem(sys.modules, 'FEAST', SimpleNamespace(__version__='1.0.5+local3d1', __file__=str(installed)))
    monkeypatch.setitem(sys.modules, 'numpy', SimpleNamespace(__version__='1.26.4'))
    monkeypatch.setitem(sys.modules, 'ot', SimpleNamespace(__version__='0.9.7'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='12.4'), cuda=SimpleNamespace(is_available=lambda: False)))
    verified = {'commit': 'recorded-build', 'checkout_head': 'later-commit', 'verified_package_files': 29}
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[1].endswith('verify_feast_install.py')
        return SimpleNamespace(stdout=json.dumps(verified))
    monkeypatch.setattr(workflow.subprocess, 'run', run)
    config = {'_feast_provenance': tmp_path / 'provenance.json', '_feast_repo': tmp_path / 'FEAST',
              'required_feast_commit': 'recorded-build', 'required_feast_version': '1.0.5+local3d1'}
    identity = workflow.feast_identity(config, require_cuda=False)
    assert identity['commit'] == 'recorded-build'
    assert len(calls) == 1
    config['required_feast_commit'] = 'different-build'
    with pytest.raises(RuntimeError, match='commit mismatch'):
        workflow.feast_identity(config, require_cuda=False)
