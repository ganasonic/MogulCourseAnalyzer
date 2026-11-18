"""Incremental analysis persistence. Raw tracks are immutable; reviews are audited."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import json
from uuid import uuid4
from .analytics import Run,summarize
from .models import Calibration
from .repository import EventRepository


def now():
    return datetime.now(timezone.utc).isoformat()


class AnalysisRepository(EventRepository):
    def begin(self,calibration,session_id,config,model_sha):
        session_id=self.save(calibration,session_id)
        identifier=str(uuid4())
        data={'calibration':calibration.to_dict(),'config':config,'model_sha256':model_sha,
              'frames_processed':0,'dropped_frames':0,'last_timestamp_ms':None,'error':None}
        with closing(self._connect()) as db,db:
            db.execute('INSERT INTO analysis_sessions VALUES(?,?,?,?,?)',(identifier,session_id,now(),'running',json.dumps(data,ensure_ascii=False)))
        return identifier

    def write_frame(self,identifier,records,runs,processed,dropped,timestamp):
        with closing(self._connect()) as db,db:
            db.executemany('INSERT OR IGNORE INTO tracks VALUES(?,?,?,?,?)',
                [(identifier,r['frame_index'],r['track_id'],r['timestamp_ms'],json.dumps(r)) for r in records])
            db.executemany('INSERT INTO runs VALUES(?,?,?) ON CONFLICT(analysis_id,id) DO UPDATE SET data_json=excluded.data_json',
                [(identifier,r['id'],json.dumps(r,ensure_ascii=False)) for r in runs])
            data=json.loads(db.execute('SELECT data_json FROM analysis_sessions WHERE id=?',(identifier,)).fetchone()[0])
            data.update(frames_processed=processed,dropped_frames=dropped,last_timestamp_ms=timestamp)
            db.execute('UPDATE analysis_sessions SET data_json=? WHERE id=?',(json.dumps(data,ensure_ascii=False),identifier))

    def finish(self,identifier,status,error=None,metrics=None):
        with closing(self._connect()) as db,db:
            data=json.loads(db.execute('SELECT data_json FROM analysis_sessions WHERE id=?',(identifier,)).fetchone()[0])
            data['error']=error
            if metrics:
                data.update(metrics)
            db.execute('UPDATE analysis_sessions SET status=?,data_json=? WHERE id=?',(status,json.dumps(data,ensure_ascii=False),identifier))

    def attach_recordings(self,identifier,paths):
        with closing(self._connect()) as db,db:
            data=json.loads(db.execute('SELECT data_json FROM analysis_sessions WHERE id=?',(identifier,)).fetchone()[0])
            data['recordings']=list(dict.fromkeys(data.get('recordings',[])+list(paths)))
            db.execute('UPDATE analysis_sessions SET data_json=? WHERE id=?',(json.dumps(data,ensure_ascii=False),identifier))

    def list_analyses(self):
        with closing(self._connect()) as db:
            exists=db.execute("SELECT name FROM sqlite_master WHERE name='analysis_sessions'").fetchone()
            if not exists:
                return []
            return [{'id':r[0],'created_at':r[1],'status':r[2]} for r in db.execute('SELECT id,created_at,status FROM analysis_sessions ORDER BY rowid DESC LIMIT 50')]

    def load_analysis(self,identifier):
        with closing(self._connect()) as db:
            row=db.execute('SELECT session_id,status,data_json FROM analysis_sessions WHERE id=?',(identifier,)).fetchone()
            if not row:
                raise ValueError('解析記録がありません')
            runs=[json.loads(r[0]) for r in db.execute('SELECT data_json FROM runs WHERE analysis_id=? ORDER BY rowid',(identifier,))]
        data=json.loads(row[2])
        data.update(id=identifier,session_id=row[0],status=row[1],runs=runs)
        return data

    def tracks_at(self,identifier,frame_index):
        with closing(self._connect()) as db:
            records=[json.loads(r[0]) for r in db.execute('SELECT data_json FROM tracks WHERE analysis_id=? AND frame_index=?',(identifier,frame_index))]
            runs={r[0]:json.loads(r[1]) for r in db.execute('SELECT id,data_json FROM runs WHERE analysis_id=?',(identifier,))}
        visible=[]
        for record in records:
            original=record['run_id']
            current=runs.get(original,{})
            if current.get('review')=='excluded':
                continue
            while current.get('merged_into'):
                current=runs[current['merged_into']]
            if current and current['id']!=original:
                record['original_run_id']=original
                record['run_id']=current['id']
            if current.get('review')!='excluded':
                visible.append(record)
        return visible

    def review(self,identifier,run_id,changes,reason):
        if not reason.strip():
            raise ValueError('修正理由を入力してください')
        data=self.load_analysis(identifier)
        if data['status']=='running':
            raise ValueError('解析停止後に修正してください')
        allowed={'qualified','start_lane','passed_lanes','review','merged_into','change_count_override','completed_lane'}
        if set(changes)-allowed:
            raise ValueError('修正できない項目です')
        runs=deepcopy(data['runs'])
        run=next((r for r in runs if r['id']==run_id),None)
        if run is None:
            raise ValueError('滑走記録がありません')
        before=deepcopy(run)
        run.update(changes)
        c=Calibration.from_dict(data['calibration'])
        lanes={l.id for l in c.lanes}
        if type(run['qualified']) is not bool or run.get('review') not in ('unreviewed','confirmed','excluded') or run.get('start_lane') not in lanes|{None} or run.get('completed_lane') not in lanes|{None} or not set(run.get('passed_lanes',[]))<=lanes:
            raise ValueError('人数・レーン・確認状態が不正です')
        override=run.get('change_count_override')
        if override is not None and (type(override) is not int or not 0<=override<=9999):
            raise ValueError('コースチェンジ件数が不正です')
        if run.get('merged_into')==run_id:
            raise ValueError('同じ滑走IDへ統合できません')
        summarize(runs,c.lanes)  # validates merge references and cycles before writing
        with closing(self._connect()) as db,db:
            db.execute('UPDATE runs SET data_json=? WHERE analysis_id=? AND id=?',(json.dumps(run,ensure_ascii=False),identifier,run_id))
            db.execute('INSERT INTO run_reviews(analysis_id,run_id,created_at,reason,before_json,after_json) VALUES(?,?,?,?,?,?)',
                       (identifier,run_id,now(),reason.strip(),json.dumps(before,ensure_ascii=False),json.dumps(run,ensure_ascii=False)))

    def review_event(self,identifier,run_id,event_id,state,reason):
        data=self.load_analysis(identifier)
        if data['status']=='running' or not reason.strip() or state not in ('confirmed','excluded','unreviewed'):
            raise ValueError('停止後に確認状態と理由を指定してください')
        run=next((r for r in data['runs'] if r['id']==run_id),None)
        if run is None:
            raise ValueError('滑走記録がありません')
        before=deepcopy(run)
        event=next((e for e in run.get('events',[]) if e['id']==event_id),None)
        if event is None:
            raise ValueError('イベントがありません')
        event['review']=state
        with closing(self._connect()) as db,db:
            db.execute('UPDATE runs SET data_json=? WHERE analysis_id=? AND id=?',(json.dumps(run,ensure_ascii=False),identifier,run_id))
            db.execute('INSERT INTO run_reviews(analysis_id,run_id,created_at,reason,before_json,after_json) VALUES(?,?,?,?,?,?)',
                       (identifier,run_id,now(),reason.strip(),json.dumps(before,ensure_ascii=False),json.dumps(run,ensure_ascii=False)))

    def add_event(self,identifier,run_id,kind,timestamp,reason):
        if not reason.strip() or kind not in ('stumble_candidate','fall_candidate','course_out','course_exit'):
            raise ValueError('イベント種別と理由を指定してください')
        data=self.load_analysis(identifier)
        if data['status']=='running':raise ValueError('分析停止後に追加してください')
        run=next((r for r in data['runs'] if r['id']==run_id),None)
        if run is None:raise ValueError('滑走を選択してください')
        with closing(self._connect()) as db:
            rows=db.execute('SELECT data_json FROM tracks WHERE analysis_id=? AND timestamp_ms BETWEEN ? AND ? ORDER BY ABS(timestamp_ms-?)',
                            (identifier,timestamp-500,timestamp+500,timestamp)).fetchall()
        by_id={r['id']:r for r in data['runs']}
        def belongs(other):
            seen=set()
            while other in by_id and other not in seen:
                if other==run_id:return True
                seen.add(other);other=by_id[other].get('merged_into')
            return False
        record=next((r for row in rows for r in [json.loads(row[0])] if belongs(r['run_id'])),None)
        if record is None:raise ValueError('指定時刻付近に選択滑走の位置記録がありません')
        before=deepcopy(run)
        run.setdefault('events',[]).append({'id':str(uuid4()),'kind':kind,'point':record['point'],
            'timestamp_ms':record['timestamp_ms'],'lane_id':record.get('lane_id') or run.get('lane'),
            'review':'confirmed','bump_id':None,'bump_label':None,'origin':'manual'})
        with closing(self._connect()) as db,db:
            db.execute('UPDATE runs SET data_json=? WHERE analysis_id=? AND id=?',(json.dumps(run,ensure_ascii=False),identifier,run_id))
            db.execute('INSERT INTO run_reviews(analysis_id,run_id,created_at,reason,before_json,after_json) VALUES(?,?,?,?,?,?)',
                       (identifier,run_id,now(),reason.strip(),json.dumps(before,ensure_ascii=False),json.dumps(run,ensure_ascii=False)))

    def add_manual(self,identifier,lane,timestamp,reason):
        if not reason.strip():
            raise ValueError('手動追加の理由を入力してください')
        data=self.load_analysis(identifier)
        c=Calibration.from_dict(data['calibration'])
        if data['status']=='running' or lane not in {l.id for l in c.lanes} or timestamp<0:
            raise ValueError('停止後に正しいレーン・時刻を指定してください')
        run=asdict(Run(str(uuid4()),timestamp,timestamp,0,0))
        run.update(qualified=True,start_lane=None,first_lane=lane,lane=lane,passed_lanes=[lane],review='confirmed',manual=True,first_depth=None,max_depth=None)
        with closing(self._connect()) as db,db:
            db.execute('INSERT INTO runs VALUES(?,?,?)',(identifier,run['id'],json.dumps(run,ensure_ascii=False)))
            db.execute('INSERT INTO run_reviews(analysis_id,run_id,created_at,reason,before_json,after_json) VALUES(?,?,?,?,?,?)',
                       (identifier,run['id'],now(),reason.strip(),None,json.dumps(run,ensure_ascii=False)))
        return run['id']
