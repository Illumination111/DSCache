import unittest
from .meminstream_scoring import score, task_macro_report


class ScoringTests(unittest.TestCase):
    def test_exact_sets_and_official_format_policy(self):
        row=dict(question_type='multiple-choice',options=['A. Apple','B. Bowl','C. Cup'],
                 correct_option=['A','C'],correct_index=[0,2])
        for response in ('A,C','C,A','[A, C]'):
            self.assertTrue(score(row,response,'skip')['correct'])
        for response in ('A','A,B,C','A. Apple and C. Cup','A because it is correct','Z',''):
            self.assertFalse(score(row,response,'skip')['correct'])
        row.update(correct_option='B',correct_index=1)
        self.assertTrue(score(row,'B. Bowl','skip')['correct'])
        self.assertFalse(score(row,'B. Bowl is here','skip')['format_valid'])

    def test_level2_macro_differs_from_micro_and_source_taxonomy(self):
        rows=[dict(level1='C',level2='frequent',original_task='same_source_task',
                   status='completed',correct=True) for _ in range(9)]
        rows.append(dict(level1='C',level2='rare',original_task='same_source_task',
                         status='completed',correct=False))
        report=task_macro_report(rows)
        self.assertEqual(report['task_count'],2)
        category=report['by_level1']['C']
        self.assertEqual(category['macro_accuracy'],.5)
        self.assertEqual(category['sample_micro']['accuracy_scored'],.9)

    def test_missing_and_failed_count_in_denominator_without_final_macro(self):
        rows=[dict(level1='C',level2='T',status='completed',correct=True),
              dict(level1='C',level2='T',status='missing_prediction',correct=None),
              dict(level1='C',level2='T',status='completed',correct=False,format_valid=False)]
        c=task_macro_report(rows)['by_level1']['C']
        self.assertIsNone(c['macro_accuracy'])
        self.assertFalse(c['complete'])
        self.assertEqual(c['tasks']['T']['accuracy'],1/3)
        self.assertEqual(c['macro_accuracy_lower_bound'],1/3)
        final=task_macro_report(rows,finalized=True)['by_level1']['C']
        self.assertEqual(final['macro_accuracy'],1/3)
        self.assertFalse(final['complete'])
        self.assertTrue(final['finalized'])

if __name__=='__main__':unittest.main()
