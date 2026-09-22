"""اختبار النماذج الفعلية: python validate_project.py (قد يستغرق دقائق على CPU)."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from rag_marimo import app

ROOT = Path(__file__).resolve().parent
cases = json.loads((ROOT / 'evaluation.json').read_text(encoding='utf-8'))

# أول فتح يجب ألا ينزّل النماذج أو يقرأ الإجابات المرجعية.
_, startup = app.run()
assert len(startup['pages']) == 152
assert 'model' not in startup and 'evaluation_cases' not in startup

# تحقق من المقاييس في حالات الغياب، وخارج k، وأكثر من مصدر صحيح.
metric = startup['retrieval_metrics']
assert metric([10, 20], {30}, 2)['RR'] == 0
assert metric([10, 20, 30], {30}, 2)['RR'] == 0
assert metric([10, 20], {20, 30}, 2) == {'Hit@1': 0., 'Recall': .5, 'RR': .5}
for case in cases:
    if case['relevant_sources']:
        assert startup['relevant_ids'](case)

start = time.perf_counter()
outputs, definitions = app.run(defs={
    'prepare': SimpleNamespace(value=True),
    'question_form': SimpleNamespace(value=cases[3]['question']),
    'evaluate_button': SimpleNamespace(value=True),
    'outside_button': SimpleNamespace(value=True),
})

# FAISS Inner Product يجب أن يطابق Cosine السابق عند تطبيع المتجهات.
import numpy as np
for case in cases[:8]:
    q = definitions['encoder'].encode('query: ' + case['question'], normalize_embeddings=True)
    expected = np.sort(definitions['embeddings'] @ q)[::-1][:5]
    actual, _ = definitions['index'].search(np.asarray([q], dtype='float32'), 5)
    assert np.allclose(expected, actual[0], atol=1e-5)

report = {
    'schema_version': 2,
    'status': 'running',
    'cases': cases,  # نسخة من الأسئلة وشرائحها تحفظ مع التشغيل نفسه.
    'tested_at': datetime.now(timezone.utc).isoformat(),
    'data_sha256': hashlib.sha256((ROOT / 'data/labor_2026.md').read_bytes()).hexdigest(),
    'settings': {key: definitions[key] for key in ['EMBEDDING_MODEL', 'RERANKER_MODEL', 'ANSWER_MODEL', 'CANDIDATE_K', 'TOP_K', 'DEVICE']},
    'counts': {'pages': len(definitions['pages']), 'blocks': len(definitions['blocks']), 'chunks': len(definitions['chunks'])},
    'summary': definitions['summary_rows'],
    'retrieval': definitions['retrieval_rows'],
    'outside': definitions['outside_rows'],
    'answers': [],
    'validation': {'startup_without_models': True, 'reference_labels_resolved': True, 'metrics_edge_cases': True, 'faiss_matches_cosine': True},
}
print('METRICS', json.dumps(report['summary'], ensure_ascii=False), flush=True)
print('OUTSIDE', json.dumps(report['outside'], ensure_ascii=False), flush=True)

def save():
    report['elapsed_seconds'] = round(time.perf_counter() - start, 2)
    (ROOT / 'validation_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

# نولّد إجابات لجميع الأسئلة داخل النطاق، بنفس الدوال المستخدمة في الواجهة.
for case in cases:
    if not case['relevant_sources']:
        continue
    before = time.perf_counter()
    candidates = definitions['retrieve'](case['question'], definitions['CANDIDATE_K'])
    ranked = definitions['rerank'](case['question'], candidates)
    hits = ranked[:definitions['TOP_K']]
    answer = definitions['answer'] if case['id'] == 4 else definitions['generate_answer'](case['question'], hits)
    row = {'id': case['id'], 'question': case['question'], 'answer': answer,
           'sources': [{'block': hit['block'], 'page': hit['page'], 'text': hit['text'], 'score': hit['score'], 'rerank_score': hit['rerank_score']} for hit in hits],
           'reference_answer': case['reference_answer'],
           'elapsed_seconds': None if case['id'] == 4 else round(time.perf_counter() - before, 2),
           'reused_answer_from_initial_notebook_run': case['id'] == 4}
    report['answers'].append(row)
    save()
    print('ANSWER', json.dumps(row, ensure_ascii=False), flush=True)

report['status'] = 'complete'
save()
print('SAVED', ROOT / 'validation_results.json', 'seconds', report['elapsed_seconds'], flush=True)
