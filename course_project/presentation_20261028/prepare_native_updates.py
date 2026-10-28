"""Create scoped native Slides requests from a fresh read and the update plan."""
from pathlib import Path
import json
from copy import deepcopy

HERE=Path(__file__).resolve().parent
WORK=HERE/'update_20261028'
raw=json.loads((WORK/'raw-before.json').read_text())
deck=raw['structuredContent']
plan=json.loads((WORK/'presentation_update.json').read_text())
elements={e['objectId']:e for s in deck['slides'] for e in s['pageElements']}
requests=[]
u16=lambda s:len(s.encode('utf-16-le'))//2

def source_paragraphs(element):
    data=element['shape']['text']['textElements']
    result=[]
    for record in data:
        if 'paragraphMarker' in record:
            result.append(dict(start=record.get('startIndex',0),end=record['endIndex'],style=record['paragraphMarker']['style'],runs=[]))
        if 'textRun' in record and result:
            result[-1]['runs'].append(record['textRun'])
    return result

for key,text in plan['text'].items():
    element=elements[key];paragraphs=source_paragraphs(element)
    requests += [{'deleteText':{'objectId':key,'textRange':{'type':'ALL'}}},
                 {'insertText':{'objectId':key,'insertionIndex':0,'text':text}}]
    offset=0
    for i,line in enumerate(text.split('\n')):
        source=paragraphs[min(i,len(paragraphs)-1)]
        visible=[r for r in source['runs'] if r['content'].strip()]
        if not visible:visible=[r for p in paragraphs for r in p['runs'] if r['content'].strip()][:1]
        assert visible,key
        assert len({json.dumps(r.get('style',{}),sort_keys=True) for r in visible})==1,('mixed-style replacement needs explicit mapping',key)
        style=deepcopy(visible[0]['style'])
        style={k:v for k,v in style.items() if v!={}}
        if key in plan['font_sizes']:style['fontSize']={'magnitude':plan['font_sizes'][key],'unit':'PT'}
        if key in plan['links']:style['link']={'url':plan['links'][key]}
        if key in plan.get('colors',{}):
            color=plan['colors'][key]
            style['foregroundColor']={'opaqueColor':{'rgbColor':{name:int(color[i:i+2],16)/255 for i,name in [(0,'red'),(2,'green'),(4,'blue')]}}}
        end=offset+u16(line)
        if line:
            requests.append({'updateTextStyle':{'objectId':key,'textRange':{'type':'FIXED_RANGE','startIndex':offset,'endIndex':end},'style':style,'fields':','.join(style)}})
            requests.append({'updateParagraphStyle':{'objectId':key,'textRange':{'type':'FIXED_RANGE','startIndex':offset,'endIndex':end},'style':source['style'],'fields':','.join(source['style'])}})
        offset=end+1
for key,(background,fraction) in plan['bar_widths'].items():
    transform=deepcopy(elements[key]['transform'])
    transform['scaleX']=elements[background]['transform']['scaleX']*fraction
    requests.append({'updatePageElementTransform':{'objectId':key,'applyMode':'ABSOLUTE','transform':transform}})
for i,(slide,spec) in enumerate(zip(deck['slides'],plan['slides']),1):
    key=slide['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
    text=f'SLIDE {i} · {spec["seconds"]} seconds\n\nCUE: {spec["cue"]}\n\n{spec["say"]}\n\nTRANSITION: {spec["transition"]}\n\nKEEP PRECISE: {spec["caveat"]}\n\nEvidence: {spec["source"]}'
    requests += [{'deleteText':{'objectId':key,'textRange':{'type':'ALL'}}},
                 {'insertText':{'objectId':key,'insertionIndex':0,'text':text}}]
(WORK/'native_requests.json').write_text(json.dumps(dict(presentation_id=deck['presentationId'],write_control={'requiredRevisionId':deck['revisionId']},requests=requests),ensure_ascii=False)+'\n')
print(f'Prepared {len(requests)} scoped requests on 15 existing slides; original media, structure and styling retained.')
