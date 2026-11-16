import unittest
from .meminstream_prompts import build_request


class PromptTests(unittest.TestCase):
    def row(self):
        return dict(question='Which object was seen?', query_time=10, question_type='multiple-choice',
                    options=['A. Apple', 'B. Bowl'], correct_index=0, correct_option='A', answer='Apple')

    def test_single_choice(self):
        r=build_request(self.row())
        self.assertFalse(r['multiple'])
        self.assertEqual(r['options'],['Apple','Bowl'])
        self.assertEqual(set(r),{'question','query_time','options','multiple'})

    def test_normalize_mislabeled_multiselect(self):
        row=self.row();row.update(correct_index=[0,1],correct_option=['A','B'])
        self.assertTrue(build_request(row)['multiple'])

    def test_label_values_do_not_change_request(self):
        row=self.row();before=build_request(row)
        row.update(correct_index=1,correct_option='B',answer='Bowl')
        self.assertEqual(build_request(row),before)
        row.update(correct_index=[0],correct_option=['A']);before=build_request(row)
        row.update(correct_index=[1,0],correct_option=['B','A'])
        self.assertEqual(build_request(row),before)

    def test_explicit_multiselect_wording(self):
        row=self.row();row['question']='请选择看到的物体（多选）'
        self.assertTrue(build_request(row)['multiple'])

if __name__=='__main__':unittest.main()
