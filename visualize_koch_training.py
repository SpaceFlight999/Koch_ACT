"""只读训练看板：四组、独立合并组和训练曲线。"""
from koch_project_paths import read_project_json
import argparse
import csv
import math
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from show_koch_layers import ROOT, TITLES

SKILLS = dict(full_task='完整搬运', reach_S1='到达 S1', reach_S2='到达 S2',
    reach_S3='到达 S3', reach_S4='到达 S4', gripper='夹爪开合', lift='夹持后提起',
    transfer='搬运到 T', lower='向下放置', release='放置后松爪', rest_lift='休息位提起',
    approach_move='移到 S 上方', approach_align='向下对齐 S', return_lift='松爪后提起',
    return_move='返回休息位', rest_align='调整休息姿态')
MERGED = 'merged'
GROUP_LABELS = {**TITLES, MERGED: '合并搬运'}
PHASES = {'merged_pretrain': ('分项辅助预训练', 'phase_pretrain', 1200),
          'merged_full': ('完整搬运微调', '', 2400)}
ACTIVE_LABELS = {'active_takeoff': '主动起步提起', 'merged_active_full': '完整搬运（无起步等待）'}
ACTION_LABELS = {**SKILLS, **{key: value[0] for key,value in PHASES.items()}, **ACTIVE_LABELS}
MERGED_STATUS = {'preparing_full':'准备完整轨迹', 'waiting_for_skills':'等待分项导出',
                 'teacher_labels':'生成分项辅助标签', 'pretraining':'分项辅助预训练',
                 'finetuning':'完整轨迹微调', 'evaluating':'离线验证与导出',
                 'complete':'已导出 1/1', 'error':'已停止'}
REFRESH_MS = 2000
STALE_SECONDS = 180
NAVY, INK, MUTED, LINE, SURFACE = '#112638', '#183246', '#506879', '#d9e3ea', '#f1f5f7'
TEAL, BLUE, RED = '#08796d', '#2464a6', '#ac3737'
FONT = 'Microsoft YaHei UI'


def read_metrics(path):
    """续训若退回检查点，舍弃 CSV 中随后被重写的旧分支。"""
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            try:
                step = int(row['step'])
                loss = float(row['loss'])
                l1 = float(row['l1'])
                val = float(row['validation_mae_degrees']) if row['validation_mae_degrees'] else None
                if step < 1 or not all(math.isfinite(x) for x in (loss, l1)):
                    continue
                if val is not None and not math.isfinite(val):
                    val = None
            except (KeyError, ValueError, TypeError):
                continue  # 写入中的最后一行可能还不完整
            while rows and rows[-1][0] >= step:
                rows.pop()
            rows.append((step, loss, l1, val))
    return rows


class TrainingReader:
    def __init__(self, batch=None):
        self.requested = Path(batch).resolve() if batch else None
        self.batch = None
        self.merged_output = None
        self.active_output = None
        self.cache = {}

    def read_cached(self, path, reader, default):
        try:
            info = path.stat()
            key = (info.st_mtime_ns, info.st_size)
        except FileNotFoundError:
            return default
        previous = self.cache.get(path)
        if previous and previous[0] == key:
            return previous[1]
        try:
            value = reader(path)
        except (OSError, ValueError, UnicodeError):
            # 轮询时若刚好碰上写入，保留上次完整快照，下次重试。
            if previous:
                return previous[1]
            return default
        self.cache[path] = (key, value)
        return value

    def json(self, path, default=None):
        return self.read_cached(path, read_project_json, default)

    def metrics(self, group, skill):
        if self.batch is None:
            return []
        if group == MERGED:
            if skill in ACTIVE_LABELS and self.active_output is not None:
                return self.read_cached(self.active_output/'models'/skill/'metrics.csv', read_metrics, [])
            if self.merged_output is None or skill not in PHASES:
                return []
            path = self.merged_output/PHASES[skill][1]/'models/merged_full/metrics.csv'
            return self.read_cached(path, read_metrics, [])
        return self.read_cached(self.batch/group/'models'/skill/'metrics.csv', read_metrics, [])

    def merged_snapshot(self, batch):
        self.merged_output = None
        self.active_output = None
        folder = ROOT/'outputs/koch_merged'
        pointer = self.json(folder/'latest_batch.json', {})
        if not pointer.get('output') or not pointer.get('source_batch') or Path(pointer['source_batch']).resolve()!=batch:
            return None
        output = Path(pointer['output']).resolve()
        if not output.is_relative_to(folder.resolve()) or not output.is_dir():
            return None
        self.merged_output = output
        path = output/'queue_state.json'
        queue = self.json(path, {})
        status = queue.get('status')
        phase = 'merged_full' if status in ('finetuning','evaluating','complete') else 'merged_pretrain'
        if status == 'error' and (output/'models/merged_full/progress.json').exists():
            phase = 'merged_full'
        index = self.json(output/'trained_models/index.json', {}).get('models', {})
        rows = {}
        files = [path]
        for name, (_, subdir, steps) in PHASES.items():
            progress = output/subdir/'models/merged_full/progress.json'
            rows[name] = {'steps': steps, **self.json(progress, {}),
                          'exported': name == 'merged_full' and name in index}
            if name == phase:
                files.extend((progress, progress.with_name('metrics.csv')))
        active_root = ROOT/'outputs/koch_active'
        active_pointer = self.json(active_root/'latest_batch.json', {})
        if active_pointer.get('source_batch') == str(batch) and active_pointer.get('output'):
            active_output = Path(active_pointer['output']).resolve()
            active_queue_path = active_output/'queue_state.json'
            if active_output.parent == active_root.resolve() and (active_queue := self.json(active_queue_path, {})):
                self.active_output = active_output
                active_index = self.json(active_output/'trained_models/index.json', {}).get('models', {})
                for name, steps in active_queue.get('steps', {}).items():
                    if name in ACTIVE_LABELS:
                        progress = active_output/'models'/name/'progress.json'
                        rows[name] = {'steps': steps, **self.json(progress, {}), 'exported': name in active_index}
                phase = active_queue.get('skill') or 'merged_active_full'
                output, queue = active_output, active_queue
                status = queue.get('status')
                progress = active_output/'models'/phase/'progress.json'
                files = [active_queue_path, progress, progress.with_name('metrics.csv')]
        updated = 0
        for file in files:
            try:
                updated = max(updated, file.stat().st_mtime)
            except FileNotFoundError:
                pass
        stale = status not in (None,'complete','error') and updated and time.time()-updated > STALE_SECONDS
        return dict(queue=queue,rows=rows,phase=phase,updated=updated,stale=bool(stale),output=output,
                    active=output == self.active_output)

    def snapshot(self):
        folder = ROOT/'outputs/koch_layers'
        pointer = self.json(folder/'latest_batch.json', {})
        batch = self.requested or (Path(pointer['batch']).resolve() if pointer.get('batch') else None)
        if batch is None or not batch.is_relative_to(folder.resolve()) or not batch.is_dir():
            raise ValueError('找不到四组训练批次。')
        if batch != self.batch:
            self.batch = batch
            self.cache.clear()
        queue_path = batch/'queue_state.json'
        queue = self.json(queue_path, {})
        active = (queue.get('group'), queue.get('skill'))
        mtime = queue_path.stat().st_mtime if queue_path.exists() else 0
        groups = {}
        for group in TITLES:
            directory = batch/group
            manifest = self.json(directory/'manifest.json', {})
            index = self.json(directory/'trained_models/index.json', {}).get('models', {})
            skills = {}
            for name in SKILLS:
                if manifest and name not in manifest.get('skills', {}):
                    continue
                progress = directory/'models'/name/'progress.json'
                data = self.json(progress, {})
                if (group, name) == active:
                    for file in (progress, progress.with_name('metrics.csv')):
                        try:
                            mtime = max(mtime, file.stat().st_mtime)
                        except FileNotFoundError:
                            pass
                skills[name] = dict(data, exported=name in index)
            groups[group] = skills
        stale = queue.get('status') in ('training', 'preparing') and mtime and time.time()-mtime > STALE_SECONDS
        merged = self.merged_snapshot(batch)
        if merged:
            groups[MERGED] = merged['rows']
        return dict(batch=batch, queue=queue, groups=groups, active=active, updated=mtime, stale=bool(stale), merged=merged)


def row_status(snapshot, group, name, item):
    if item['exported']:
        return '已导出', 'done'
    status = item.get('status')
    if group == MERGED:
        merged = snapshot['merged']
        phase_status = merged['queue'].get('status')
        if name == 'merged_pretrain' and status == 'complete':
            return '阶段完成', 'done'
        if name == merged['phase']:
            if merged['stale'] or phase_status == 'error' or status in ('error','interrupted'):
                return '待续训', 'error'
            if phase_status == 'waiting_for_skills':
                return '等待分项', 'pending'
            if phase_status in ('preparing','preparing_full','teacher_labels'):
                return '准备数据', 'running'
            if phase_status in ('training','pretraining','finetuning') and status != 'complete':
                return '训练中', 'running'
            if phase_status == 'evaluating':
                return '验证导出', 'running'
    if status in ('error', 'interrupted'):
        return ('失败' if status == 'error' else '待续训'), 'error'
    if status == 'complete':
        return '待导出', 'pending'
    if status == 'running':
        if snapshot['active'] == (group, name) and not snapshot['stale'] and snapshot['queue'].get('status') == 'training':
            return '训练中', 'running'
        return '待续训', 'error'
    return '待训练', 'pending'


class Curve(tk.Canvas):
    def __init__(self, parent, color):
        super().__init__(parent, bg='white', highlightthickness=0, height=160)
        self.color, self.points = color, []
        self.xmax, self.unit = 1, ''
        self.bind('<Configure>', lambda _: self.draw())
        self.bind('<Motion>', self.hover)
        self.bind('<Leave>', lambda _: self.delete('hover'))

    def set_data(self, points, xmax, unit):
        if self.points != points or self.xmax != xmax or self.unit != unit:
            self.points, self.xmax, self.unit = points, xmax, unit
            self.draw()

    def draw(self):
        self.delete('all')
        w, h = self.winfo_width(), self.winfo_height()
        if w < 130 or h < 80:
            return
        if not self.points:
            self.create_text(w/2,h/2,text='暂无数据，开始训练后自动显示',fill=MUTED,font=(FONT,10))
            return
        left, right, top, bottom = 55, w-20, 14, h-28
        ymax = max(y for _,y in self.points) * 1.06 or 1
        xmax = max(1, self.xmax, self.points[-1][0])
        self.plot = (left,right,top,bottom,xmax,ymax)
        for n in range(5):
            f=n/4
            y=bottom-(bottom-top)*f
            self.create_line(left,y,right,y,fill=LINE)
            self.create_text(left-8,y,text=f'{ymax*f:.3g}',anchor='e',fill=MUTED,font=('Consolas',9))
            x=left+(right-left)*f
            self.create_text(x,bottom+14,text=str(round(xmax*f)),fill=MUTED,font=('Consolas',9))
        coords=[self.xy(step,value) for step,value in self.points]
        if len(coords)>1:
            self.create_line(*[c for pair in coords for c in pair],fill=self.color,width=2)
        if len(coords)<=18:
            for x,y in coords:
                self.create_oval(x-3,y-3,x+3,y+3,fill=self.color,outline='white')

    def xy(self, step, value):
        l,r,t,b,xmax,ymax = self.plot
        return l+(r-l)*step/xmax, b-(b-t)*value/ymax

    def hover(self, event):
        self.delete('hover')
        if not self.points or not hasattr(self,'plot'):
            return
        left,right,top,bottom,xmax,_=self.plot
        if not left<=event.x<=right or not top<=event.y<=bottom:
            return
        step,value=min(self.points,key=lambda p:abs(p[0]-(event.x-left)*xmax/(right-left)))
        x,y=self.xy(step,value)
        self.create_line(x,top,x,bottom,fill=MUTED,dash=(2,3),tags='hover')
        self.create_oval(x-4,y-4,x+4,y+4,fill=self.color,outline='white',tags='hover')
        label=f'{step} 步  ·  {value:.3g}{self.unit}'
        self.create_text(min(max(x,left+75),right-75),top+9,text=label,fill=INK,font=(FONT,9),tags='hover')


class Dashboard:
    def __init__(self, root, reader):
        self.root,self.reader=root,reader
        self.group,self.skill=next(iter(TITLES)), 'full_task'
        self.snapshot=None
        self.follow=tk.BooleanVar(root,True)
        self.loss_kind=tk.StringVar(root,'L1 损失')
        root.title('Koch · 训练可视化')
        root.geometry('1120x740')
        root.minsize(910,640)
        root.configure(bg=SURFACE)
        style=ttk.Style(root)
        style.theme_use('clam')
        style.configure('.',font=(FONT,10))
        style.configure('Treeview',rowheight=25,background='white',fieldbackground='white',foreground=INK)
        style.configure('Treeview.Heading',font=(FONT,10,'bold'),padding=6)
        style.configure('Koch.Horizontal.TProgressbar',background=TEAL,troughcolor=LINE,borderwidth=0)
        header=tk.Frame(root,bg=NAVY,padx=22,pady=14)
        header.pack(fill='x')
        tk.Label(header,text='Koch 训练看板',font=(FONT,22,'bold'),bg=NAVY,fg='white').pack(anchor='w')
        self.summary=tk.Label(header,text='正在读取训练记录…',bg=NAVY,fg='#d4e6ef',font=(FONT,10),anchor='w')
        self.summary.pack(fill='x',pady=(5,0))
        strip=tk.Frame(root,bg=SURFACE,padx=14,pady=12)
        strip.pack(fill='x')
        self.group_views={}
        self.group_strip = strip
        for column,(group,title) in enumerate(GROUP_LABELS.items()):
            strip.grid_columnconfigure(column,weight=1,uniform='group')
            panel=tk.Frame(strip,bg='white',padx=10,pady=7,highlightbackground=LINE,highlightthickness=1)
            panel.grid(row=0,column=column,sticky='ew',padx=4)
            button=ttk.Button(panel,text=title,command=lambda g=group:self.choose_group(g))
            button.pack(fill='x')
            detail=tk.Label(panel,text='—',bg='white',fg=MUTED,font=(FONT,9))
            detail.pack(anchor='w',pady=(4,3))
            bar=ttk.Progressbar(panel,style='Koch.Horizontal.TProgressbar',maximum=16)
            bar.pack(fill='x')
            self.group_views[group]=(panel,detail,bar)
        body=tk.Frame(root,bg=SURFACE,padx=18,pady=3)
        body.pack(fill='both',expand=True)
        left=tk.Frame(body,bg='white',padx=10,pady=8)
        left.pack(side='left',fill='y',padx=(0,10))
        right=tk.Frame(body,bg='white',padx=12,pady=8)
        right.pack(side='left',fill='both',expand=True)
        self.group_label=tk.Label(left,bg='white',fg=INK,font=(FONT,12,'bold'),anchor='w')
        self.group_label.pack(fill='x',pady=(0,5))
        ttk.Checkbutton(left,text='跟随当前训练模型',variable=self.follow,command=self.refresh).pack(anchor='w',pady=(0,7))
        table=tk.Frame(left,bg='white');table.pack(fill='both',expand=True)
        self.tree=ttk.Treeview(table,columns=('skill','progress','status'),show='headings',selectmode='browse',height=16)
        for column,label,width in [('skill','动作',136),('progress','步数',84),('status','状态',78)]:
            self.tree.heading(column,text=label)
            self.tree.column(column,width=width,anchor='w' if column=='skill' else 'center',stretch=False)
        scrollbar=ttk.Scrollbar(table,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right',fill='y');self.tree.pack(side='left',fill='both',expand=True)
        for name,color in [('done',TEAL),('running',BLUE),('error',RED),('pending',MUTED)]:
            self.tree.tag_configure(name,foreground=color)
        self.tree.bind('<<TreeviewSelect>>',self.select_skill)
        self.detail=tk.Label(right,bg='white',fg=INK,font=(FONT,13,'bold'),anchor='w')
        self.detail.pack(fill='x')
        self.stats=tk.Label(right,bg='white',fg=MUTED,font=(FONT,10),anchor='w')
        self.stats.pack(fill='x',pady=(5,8))
        toolbar=tk.Frame(right,bg='white');toolbar.pack(fill='x')
        tk.Label(toolbar,text='训练损失 · 25 步均值',bg='white',fg=INK,font=(FONT,10,'bold')).pack(side='left')
        loss=ttk.Combobox(toolbar,textvariable=self.loss_kind,values=('L1 损失','总损失'),state='readonly',width=12)
        loss.pack(side='right');loss.bind('<<ComboboxSelected>>',lambda _:self.render_detail())
        self.loss=Curve(right,TEAL);self.loss.pack(fill='both',expand=True)
        tk.Label(right,text='验证角度误差（°）',bg='white',fg=INK,font=(FONT,10,'bold'),anchor='w').pack(fill='x',pady=(5,0))
        self.validation=Curve(right,BLUE);self.validation.pack(fill='both',expand=True)
        self.footer=tk.Label(root,bg=SURFACE,fg=MUTED,font=(FONT,9),anchor='w',padx=18,pady=6)
        self.footer.pack(fill='x')
        self.tick()

    def choose_group(self, group):
        if not self.snapshot or group not in self.snapshot['groups']:
            return
        self.follow.set(False)
        self.group=group
        self.refresh()

    def select_skill(self, _):
        selected=self.tree.selection()
        if selected and selected[0]!=self.skill:
            self.skill=selected[0]
            self.follow.set(False)
            self.render_detail()

    def tick(self):
        self.refresh()
        self.root.after(REFRESH_MS,self.tick)

    def refresh(self):
        try:
            self.snapshot=s=self.reader.snapshot()
        except (OSError, ValueError, KeyError) as exc:
            self.summary.configure(text='无法读取训练记录 · '+str(exc))
            return
        groups=s['groups'];queue=s['queue'];ag,an=s['active']
        merged=s['merged']
        total=sum(len(groups[g]) for g in TITLES)
        done=sum(x['exported'] for g in TITLES for x in groups[g].values())
        queue_labels={'training':'训练中','preparing':'准备数据','complete':'已完成','error':'已停止','interrupted':'已中断'}
        label='进度超过 3 分钟未更新' if s['stale'] else queue_labels.get(queue.get('status'),'等待记录')
        current=f' · {TITLES[ag]} / {SKILLS.get(an,an or "准备数据")}' if ag in TITLES else ''
        self.summary.configure(text=f'{label}   |   已验证导出 {done}/{total}   |   剩余 {total-done}{current}')
        merged_panel,merged_detail,merged_bar=self.group_views[MERGED]
        self.group_strip.grid_columnconfigure(len(TITLES),weight=int(bool(merged)),uniform='group' if merged else '')
        if merged:
            merged_panel.grid()
            q=merged['queue']; status=q.get('status')
            text=MERGED_STATUS.get(status,'等待记录')
            if merged['active']:
                n=sum(merged['rows'][name]['exported'] for name in ACTIVE_LABELS if name in merged['rows'])
                text=f'无等待模型 {n}/2' + (' · 训练中' if status == 'training' else ' · '+text if status != 'complete' else '')
            if status=='waiting_for_skills':
                text=f'等待分项 {done}/{total}'
            if merged['stale']:
                text='进度未更新'
            merged_detail.configure(text=text)
            merged_bar.configure(maximum=sum(r['steps'] for r in merged['rows'].values()),
                                 value=sum(min(r.get('step',0),r['steps']) for r in merged['rows'].values()))
        else:
            merged_panel.grid_remove()
        if self.group not in groups:
            self.group=next(iter(TITLES))
        if self.follow.get() and merged and queue.get('status')=='complete':
            ag,an=MERGED,merged['phase']
        if self.follow.get() and ag in groups:
            self.group=ag
            if an in groups[ag]:self.skill=an
        for g in TITLES:
            skills=groups[g]
            panel,detail,bar=self.group_views[g]
            n=sum(x['exported'] for x in skills.values())
            detail.configure(text=f'已导出 {n}/{len(skills)}' + ('  ·  当前队列' if g==ag and queue.get('status') in ('training','preparing') else ''))
            bar.configure(maximum=max(1,len(skills)),value=n)
            panel.configure(highlightbackground=TEAL if g==self.group else LINE)
        merged_panel.configure(highlightbackground=TEAL if self.group==MERGED else LINE)
        self.group_label.configure(text=GROUP_LABELS[self.group])
        skills=groups[self.group]
        if self.skill not in skills:self.skill=next(iter(skills),'full_task')
        for name in self.tree.get_children():
            if name not in skills:self.tree.delete(name)
        for i,(name,item) in enumerate(skills.items()):
            text,tag=row_status(s,self.group,name,item)
            step=item.get('step',0); steps=item.get('steps',queue.get('steps_per_model',2000))
            values=(ACTION_LABELS[name],f'{step}/{steps}',text)
            if self.tree.exists(name):self.tree.item(name,values=values,tags=(tag,))
            else:self.tree.insert('',i,iid=name,values=values,tags=(tag,))
        if self.tree.exists(self.skill) and self.tree.selection()!=(self.skill,):
            self.tree.selection_set(self.skill);self.tree.see(self.skill)
        selected_queue,updated,batch=(merged['queue'],merged['updated'],merged['output']) if self.group==MERGED else (queue,s['updated'],s['batch'])
        stamp=datetime.fromtimestamp(updated).strftime('%m-%d %H:%M:%S') if updated else '—'
        reason=f'  |  {selected_queue["reason"][:125]}' if selected_queue.get('reason') else ''
        self.footer.configure(text=f'批次 {batch.name}   |   最近写入 {stamp}   |   每 2 秒刷新{reason}')
        self.render_detail()

    def render_detail(self):
        if not self.snapshot:return
        item=self.snapshot['groups'][self.group].get(self.skill,{})
        self.detail.configure(text=f'{ACTION_LABELS[self.skill]}   /   {self.skill}')
        rows=self.reader.metrics(self.group,self.skill)
        steps=item.get('steps',self.snapshot['queue'].get('steps_per_model',2000))
        step=item.get('step',0)
        mae=item.get('best_validation_mae_degrees')
        best=f'{mae:.3f}°' if isinstance(mae,(int,float)) and math.isfinite(mae) else '—'
        device=item.get('device','—').upper()
        self.stats.configure(text=f'{step}/{steps} 步   ·   最佳验证误差 {best}   ·   {device}')
        which=2 if self.loss_kind.get()=='L1 损失' else 1
        # 用滑动窗口呈现趋势；只在显示端计算，不改训练日志。
        loss=[]
        running=0.
        for i,row in enumerate(rows):
            running+=row[which]
            if i>=25:running-=rows[i-25][which]
            loss.append((row[0],running/min(i+1,25)))
        stride=max(1,len(loss)//800)
        compact=loss[::stride]
        if loss and compact[-1]!=loss[-1]:compact.append(loss[-1])
        self.loss.set_data(compact,steps,'')
        self.validation.set_data([(r[0],r[3]) for r in rows if r[3] is not None],steps,'°')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,help='指定 outputs/koch_layers 下的批次目录')
    args=parser.parse_args()
    root=tk.Tk()
    Dashboard(root,TrainingReader(args.batch))
    root.mainloop()


if __name__=='__main__':
    main()
