"""اختبار إضافات Lab 6 من سجل التشغيل الحقيقي، دون إعادة تحميل النماذج."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from rag_marimo import app

ROOT = Path(__file__).resolve().parent
outputs, definitions = app.run(defs={'report_button': SimpleNamespace(value=True)})
assert 'encoder' not in definitions and 'model' not in definitions
rows = definitions['saved_rows']
assert len(rows) == 10
assert all(review == 'لم يُراجع' for review in definitions['user_reviews'])
assert all(row['راجعها المستخدم'] == 0 and row['اجتازت المراجعة'] is None for row in definitions['review_summary'])
assert all(row['إجابات صحيحة / مراجعة المستخدم'] == 'لم تُراجع' for row in definitions['slice_rows'])

# مقارنة متوقعة مستقلة مستمدة من سجلات ترتيب المصادر التي روجعت سابقًا.
slices = {row['الفئة']: row for row in definitions['slice_rows']}
assert (slices['جدول']['المصدر الأول قبل'], slices['جدول']['المصدر الأول بعد']) == ('1/3', '2/3')
assert (slices['نص سردي']['المصدر الأول قبل'], slices['نص سردي']['المصدر الأول بعد']) == ('4/5', '5/5')
assert (slices['مباشر']['المصدر الأول قبل'], slices['مباشر']['المصدر الأول بعد']) == ('4/6', '5/6')
assert slices['صياغة بديلة']['المصدر الأول بعد'] == '2/2'
assert [row['Hit@1'] for row in definitions['final_metrics']] == [.625, .875]
errors = {(row['السؤال'], row['النوع'], row['أساس الحكم']) for row in definitions['error_details']}
assert errors == {(3, 'إضافة غير مدعومة', 'تقييم أولي للمساعد'), (7, 'ناقصة أو ملتبسة', 'تقييم أولي للمساعد')}

# لا توجد أخطاء بحث آلية في هذا التشغيل، فنختبر التصنيفين الآليين على نسخة معدّلة من سؤال واحد.
probe = copy.deepcopy(rows[0])
probe['preliminary'] = {}
probe['selected'] = [-1]
assert definitions['classify_errors']([probe], ['لم يُراجع'])[0]['النوع'] == 'المصدر لم يُختر للإجابة'
probe['before'] = {**probe['before'], 'المرشحون': [-1], 'المصادر بالترتيب': [-1]}
assert definitions['classify_errors']([probe], ['لم يُراجع'])[0]['النوع'] == 'غياب المصدر عن المرشحين'

# تشغيل جديد بلا مراجعات أولية يجب ألا يرث أحكام الإجابات القديمة.
fresh_rows = copy.deepcopy(rows)
for row in fresh_rows:
    row['preliminary'] = {}
fresh_errors = definitions['classify_errors'](fresh_rows, ['لم يُراجع'] * len(rows))
assert fresh_errors == []

# مدخلات اختبار للواجهة فقط، وليست مراجعات بشرية فعلية أو نتائج تُنشر.
feedback = ['لم يُراجع'] * len(rows)
feedback[0] = 'صحيحة ومكتملة'
feedback[2] = 'إضافة غير مدعومة'
feedback[8] = 'امتناع مناسب خارج النطاق'
_, reviewed = app.run(defs={'report_button': SimpleNamespace(value=True), 'review_form': SimpleNamespace(value=feedback)})
assert reviewed['review_summary'][0]['راجعها المستخدم'] == 2
assert reviewed['review_summary'][0]['اجتازت المراجعة'] == 1
assert reviewed['review_summary'][1]['راجعها المستخدم'] == 1
assert reviewed['review_summary'][1]['اجتازت المراجعة'] == 1
assert any(item['السؤال'] == 3 and item['أساس الحكم'] == 'مراجعة المستخدم' for item in reviewed['error_details'])
assert not any(item['السؤال'] == 3 and item['أساس الحكم'] == 'تقييم أولي للمساعد' for item in reviewed['error_details'])

# نختبر الفصل بين المراجعات عند إعادة فتح التقرير دون قيم نموذج محفوظة.
_, reset = app.run(defs={'report_button': SimpleNamespace(value=True)})
assert all(row['راجعها المستخدم'] == 0 for row in reset['review_summary'])
assert json.loads(json.dumps(reset['final_report'], ensure_ascii=False))['tested_at'] == definitions['saved_report']['tested_at']

print(json.dumps({'status': 'passed', 'cases': len(rows), 'slices': definitions['slice_rows'],
                  'error_counts': definitions['error_counts'], 'models_loaded': False,
                  'review_input_and_reset_checks': True, 'visual_browser_review': False}, ensure_ascii=False, indent=2))
