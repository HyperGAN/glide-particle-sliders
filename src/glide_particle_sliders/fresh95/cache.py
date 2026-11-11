"""Completed original PreparedReadout data only; never encoder/teacher collection."""
from pathlib import Path
import torch
from glide_particle_sliders.binding import PreparedReadout,ReadoutRow,HIDDEN,ACTIVE
from glide_particle_sliders.binding_normalized_v3 import train_statistics
from . import PROFILE,ownership
from .util import equal,canonical,read_packet,save_packet,tree,source_record
SCHEMA='GLiDE-completed-PreparedReadout-explicit-fresh95-v1'

def file_record(value):
    if(not isinstance(value,dict)or set(value)!={'path','bytes','sha256'}or not isinstance(value['path'],str)
            or not Path(value['path']).is_absolute()or'..'in Path(value['path']).parts
            or type(value['bytes'])is not int or value['bytes']<1 or not isinstance(value['sha256'],str)
            or len(value['sha256'])!=64 or any(c not in'0123456789abcdef'for c in value['sha256'])):
        raise ValueError('complete original whole-file Source/output record required')

def validate(rows,splits,origins):
    if(not isinstance(rows,(list,tuple))or len(rows)<2 or not isinstance(splits,list)or len(splits)!=len(rows)
            or any(s not in('train','dev','test')for s in splits)or splits.count('train')<2
            or not isinstance(origins,list)or len(origins)!=len(rows)):
        raise ValueError('completed native rows/splits/origins with at least two TRAIN rows required')
    ids=set()
    for item,origin in zip(rows,origins):
        if(type(item)is not PreparedReadout or type(item.row)is not ReadoutRow or not isinstance(item.row.row_id,str)
                or not item.row.row_id or item.row.row_id in ids or item.row.scale!=1.):
            raise ValueError('exact original completed PreparedReadout with unique native row ID required')
        ids.add(item.row.row_id)
        for tensor,shape in((item.row.hidden,(1,HIDDEN)),(item.target,(1,ACTIVE)),(item.neutral,(1,ACTIVE))):
            if(not isinstance(tensor,torch.Tensor)or tuple(tensor.shape)!=shape or tensor.dtype!=torch.float32
                    or tensor.device.type!='cpu'or tensor.requires_grad or not torch.isfinite(tensor).all()):
                raise ValueError('whole frozen finite CPU FP32 native hidden/logits required')
        if(not isinstance(origin,dict)or origin.get('complete_prepared_output')is not True
                or origin.get('row_id')!=item.row.row_id or origin.get('scope')not in('original-completed','synthetic-software-only')
                or origin.get('target_semantics')not in('native-positive-logits','declared-synthetic-preference-logits','synthetic-software-only')
                or not origin.get('whole_output_files')or not origin.get('source_files')):
            raise ValueError('honest completed origin and explicit native/synthetic target semantics required')
        for record in origin['whole_output_files']+origin['source_files']:file_record(record)
    canonical(origins)

def dump(item):return dict(hidden=tree(item.row.hidden,'cpu'),row_id=item.row.row_id,scale=item.row.scale,target=tree(item.target,'cpu'),neutral=tree(item.neutral,'cpu'))
def restore(value):
    if not isinstance(value,dict)or set(value)!={'hidden','row_id','scale','target','neutral'}:raise ValueError('exact completed row state required')
    return PreparedReadout(ReadoutRow(value['hidden'],value['row_id'],value['scale']),value['target'],value['neutral'])
def statistics(rows,splits):return train_statistics(dict(neutral_hidden=torch.cat([r.row.hidden for r in rows]),splits=splits))

def save_prepared_cache(path,*,prepared,splits,teacher_origins,model_identity,cache_id,owner_admission):
    ownership.guard();validate(prepared,splits,teacher_origins)
    if not isinstance(cache_id,str)or not cache_id.strip()or not callable(owner_admission):raise ValueError('cache ID and authenticating owner required')
    from .api import identity_valid
    identity_valid(model_identity)
    request=dict(operation='serialize-completed-readout',profile=PROFILE,cache_id=cache_id,model_identity=model_identity,
        teacher_origins=teacher_origins,new_encoder_or_teacher_calls=0)
    if owner_admission(request)is not True:raise ValueError('owner did not authenticate original completed data/readout origins')
    center,rms=statistics(prepared,splits)
    return save_packet(path,dict(schema=SCHEMA,profile=PROFILE,cache_id=cache_id,model_identity=model_identity,
        original_sources=ownership.sources()['original'],serializer=source_record(__file__),teacher_origins=teacher_origins,
        splits=splits,rows=[dump(r)for r in prepared],center=center,rms=rms,complete=True,new_encoder_or_teacher_calls=0))

def load_cache(pin,model_identity):
    value=read_packet(pin)
    keys={'schema','profile','cache_id','model_identity','original_sources','serializer','teacher_origins','splits','rows','center','rms','complete','new_encoder_or_teacher_calls'}
    if(not isinstance(value,dict)or set(value)!=keys or value['schema']!=SCHEMA or value['profile']!=PROFILE
            or value['model_identity']!=model_identity or value['original_sources']!=ownership.sources()['original']
            or value['serializer']!=source_record(__file__)or value['complete']is not True or value['new_encoder_or_teacher_calls']!=0):
        raise ValueError('exact completed cache and declared native Source/model identity required')
    rows=[restore(x)for x in value['rows']];validate(rows,value['splits'],value['teacher_origins'])
    center,rms=statistics(rows,value['splits'])
    if not equal(center,value['center'])or not equal(rms,value['rms']):raise ValueError('original TRAIN-only statistics differ')
    ident={k:value[k]for k in('schema','profile','cache_id','model_identity','original_sources','serializer','teacher_origins','splits')}
    ident.update(whole_file=dict(path=pin.path,bytes=pin.bytes,sha256=pin.sha256),row_count=len(rows))
    return rows,value['splits'],center,rms,ident
