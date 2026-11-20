"""Locate files in a Koch project without depending on the working directory."""
import json
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parent
PROJECT_DIRS = {'datasets', 'outputs', 'calibration', 'configs'}


def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    windows = PureWindowsPath(value)
    if windows.is_absolute():
        return path
    return ROOT / str(value).replace('\\', '/')


def _read_reference(value):
    normalized = value.replace('\\', '/')
    prefix, slash, _ = normalized.partition('/')
    if normalized.startswith('./') or (slash and prefix in PROJECT_DIRS):
        return str(project_path(normalized))
    path = Path(value)
    windows = PureWindowsPath(value)
    if not (path.is_absolute() or windows.is_absolute()):
        return value
    if path.is_absolute() and path.is_relative_to(ROOT):
        return str(path)
    # Existing project layout identifies a movable legacy path. External paths
    # without a matching local file remain absolute.
    parts = windows.parts if windows.is_absolute() else path.parts
    for index, part in enumerate(parts[:-1]):
        if part in PROJECT_DIRS:
            local = ROOT.joinpath(*parts[index:])
            if local.exists():
                return str(local)
    return value


def _write_reference(value):
    path = Path(value)
    if path.is_absolute() and path.is_relative_to(ROOT):
        return './' + path.relative_to(ROOT).as_posix()
    return value


def _map_references(obj, convert):
    if isinstance(obj, str):
        return convert(obj)
    if isinstance(obj, list):
        return [_map_references(item, convert) for item in obj]
    if isinstance(obj, dict):
        return {convert(key): _map_references(item, convert) for key, item in obj.items()}
    return obj


def portable_references(obj):
    return _map_references(obj, _write_reference)


def read_project_json(path):
    with project_path(path).open(encoding='utf-8-sig') as stream:
        return _map_references(json.load(stream), _read_reference)


def resolve_config_reference(value, config_file):
    """Resolve paths beside a config file, with support for existing project paths."""
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute():
        return Path(_read_reference(str(value)))
    beside_config = (Path(config_file).resolve().parent / path).resolve()
    if beside_config.exists():
        return beside_config
    parts = path.parts
    if parts and parts[0] in PROJECT_DIRS:
        in_project = project_path(path)
        if in_project.exists():
            return in_project
    return beside_config
