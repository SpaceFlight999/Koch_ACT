"""Fit the existing S-area HSV/Lab color model and score labelled placements."""

import copy
from collections import Counter

from camera_vision import fit_sample, read_image
from .vision import SourceReader, warp_quad


def fit_palette(config, base, records, directory):
    palette = copy.deepcopy(base)
    palette['calibration_required'] = False
    coverage = Counter()
    skipped = []
    for record in records:
        path = (directory / record['image']).resolve()
        frame = read_image(path)
        if list(frame.shape[1::-1]) != config['frame_size']:
            raise ValueError(f'照片分辨率与标定不符：{path}')
        faces = record.get('faces', config['faces'])
        for face, color in record['labels'].items():
            if faces is None:
                skipped.append({'id': record['id'], 'face': face, 'color': color, 'reason': '照片尚未对齐'})
                continue
            patch, _ = warp_quad(frame, faces[face], 64)
            try:
                profile = fit_sample(patch, min_saturation=palette.get('sample_min_saturation', 45))
            except ValueError as exc:
                skipped.append({'id': record['id'], 'face': face, 'color': color, 'reason': str(exc)})
                continue
            profile.update(source_image=str(path), source_face=face, source_quad_px=faces[face])
            palette['colors'][color]['samples'].append(profile)
            coverage[face, color] += 1
    return palette, coverage, skipped


def evaluate(config, palette, records, directory):
    reader = SourceReader(config, palette)
    counts = {'correct': 0, 'wrong': 0, 'unknown': 0}
    differences = []
    complete = 0
    for record in records:
        path = (directory / record['image']).resolve()
        if record.get('faces', config['faces']) is None:
            faces = dict.fromkeys(record['labels'])
        else:
            shot_reader = SourceReader({**config, 'faces': record['faces']}, palette) if 'faces' in record else reader
            result = shot_reader.read(read_image(path))
            faces = {item['id']: item['color'] for item in result['faces']}
        passed = True
        for face, expected in record['labels'].items():
            color = faces[face]
            if color == expected:
                counts['correct'] += 1
            else:
                passed = False
                counts['unknown' if color is None else 'wrong'] += 1
                differences.append({'id': record['id'], 'face': face, 'expected': expected, 'observed': color})
        complete += passed
    return {**counts, 'total': sum(counts.values()), 'complete_layouts': complete,
            'layout_count': len(records), 'differences': differences}
