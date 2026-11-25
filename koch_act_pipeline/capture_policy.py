"""训练和模型调用的采集日期门槛；原始示范与回放不受影响。"""
import json
from datetime import date, datetime, timedelta, timezone
from koch_robot.paths import project_path


MIN_CAPTURE_DATE = date(2026, 11, 2)
SHANGHAI = timezone(timedelta(hours=8))


def check_capture_dates(clips, cache=None):
    """核对模型使用的示范；日期早于门槛或无法核实均拒绝使用。"""
    sources = {c.get('source') for c in clips}
    if not sources or None in sources:
        raise ValueError('训练示范来源缺失，无法核实采集日期。')
    cache = {} if cache is None else cache
    for source in sorted(sources):
        path = project_path(source)
        if source not in cache:
            meta = path / 'episode.json'
            try:
                stamp = datetime.fromisoformat(json.loads(meta.read_text(encoding='utf-8'))['created_at'])
                if stamp.tzinfo is None:
                    raise ValueError('采集时间缺少时区')
                cache[source] = stamp.astimezone(SHANGHAI).date()
            except FileNotFoundError as exc:
                # 旧示范已归档时，目录日期只用于识别并拒绝旧记录，不作为放行依据。
                try:
                    day = datetime.strptime(path.name[:8], '%Y%m%d').date()
                except ValueError:
                    day = None
                if day is None or day >= MIN_CAPTURE_DATE:
                    raise ValueError(f'找不到采集元数据：{path.name}，不能核实训练日期。') from exc
                cache[source] = day
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'采集日期无法核实：{path.name}。') from exc
        day = cache[source]
        if day < MIN_CAPTURE_DATE:
            raise ValueError(f'使用了 {day} 的示范，早于 {MIN_CAPTURE_DATE}，已屏蔽。')
