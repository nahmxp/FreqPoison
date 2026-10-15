import json,re,jiwer
from corpus import DOCS
from num2words import num2words
from collections import defaultdict
def norm(t):
  t=t.lower().replace('-',' ')
  t=re.sub(r'(\d+(\.\d+)?)', lambda m: ' '+num2words(float(m.group(1)) if '.' in m.group(1) else int(m.group(1)))+' ', t)
  t=re.sub(r'[^a-z0-9 ]',' ',t); return ' '.join(w for w in t.split() if w!='and')
rs=[json.loads(l) for l in open('results/clips.jsonl')]
g=defaultdict(list)
for r in rs: g[(r['cond'],r['model'])].append(r)
for (c,m),v in g.items():
  v=[r for r in v if r['voice'] in ('lessac','ryan')]
  if not v: continue
  refs=[norm(DOCS[r['doc']][2]) for r in v]; hyps=[norm(r['hyp']) for r in v]
  print(f"{c:8s} {m:9s} n={len(v):3d} WER {jiwer.wer(refs,hyps):.3f}  STOI {sum(r.get('stoi',0) or 0 for r in v)/len(v):.2f}")
