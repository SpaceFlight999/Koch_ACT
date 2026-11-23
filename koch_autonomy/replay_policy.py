"""Read and edit the site's Replay exclusions."""
from pathlib import Path

from .planner import SLOTS, TARGETS
from .replay_bridge import DATA_ROOT, LAYER_DIRS
from .reverse import read_json, save_json


def data_and_exclusions(site_path, site):
    data = (site_path.resolve().parent / site.get('data', DATA_ROOT)).resolve()
    return data, {(data / item).resolve() for item in site.get('excluded_replay_paths', ())}


def route_folders(data, pair, route):
    group = data / LAYER_DIRS[pair]
    return {(group / name).resolve() for name in (route, route.replace('_', '-'))}


def route_is_disabled(site_path, source, target, pair):
    site = read_json(site_path)
    data, excluded = data_and_exclusions(site_path, site)
    return bool(route_folders(data, pair, f'{source}_{target}') & excluded)


def toggle_route(site_path, source, target, pair):
    if source not in SLOTS or target not in TARGETS or pair not in LAYER_DIRS:
        raise ValueError('请选择有效的 S/T 位置和层位。')
    site = read_json(site_path)
    data, excluded = data_and_exclusions(site_path, site)
    route = f'{source}_{target}'
    folders = route_folders(data, pair, route)
    disabled = not bool(folders & excluded)
    entries = site.get('excluded_replay_paths', [])
    if disabled:
        entries = [*entries, (Path(LAYER_DIRS[pair]) / route).as_posix()]
    else:
        entries = [item for item in entries if (data / item).resolve() not in folders]
    site['excluded_replay_paths'] = entries
    save_json(site_path, site)
    return disabled
