import copy
import unittest
from .meminstream_prompts import build_request
from .meminstream_official_prompt import make_prompt


class PromptTests(unittest.TestCase):
    def row(self):
        return dict(question='Which object was seen?', query_time=10, question_type='multiple-choice',
                    options=['A. Apple', 'B. Bowl'], correct_index=0, correct_option='A', answer='Apple')

    def test_exact_released_prompt_and_unknown_cardinality(self):
        row=self.row(); request=build_request(row)
        self.assertTrue(request['multiple'])
        self.assertEqual(request['options'],['Apple','Bowl'])
        self.assertEqual(request['prompt'],make_prompt(row,0.,10.))
        self.assertIn('Select all correct options.', request['prompt'])
        self.assertNotIn('only the correct option letter.', request['prompt'])

    def test_labels_types_lengths_and_metadata_never_change_request(self):
        row=self.row();before=build_request(row)
        for labels in ({'correct_index':1,'correct_option':'B','answer':'Bowl'},
                       {'correct_index':[0],'correct_option':['A']},
                       {'correct_index':[0,1],'correct_option':['A','B']},
                       {'correct_index':None,'correct_option':None,'answer':'SECRET',
                        'level1':'SECRET','evidence':[[99,100]],'QA_type':'mc_multi'}):
            updated=copy.deepcopy(row);updated.update(labels)
            self.assertEqual(build_request(updated),before)
        stripped={k:v for k,v in row.items() if k not in ('correct_index','correct_option','answer')}
        self.assertEqual(build_request(stripped),before)

    def test_dialogue_is_strictly_past_and_supports_both_formats(self):
        row=self.row()
        row['dialogue_events']=[{'time':1,'role':'user','content':'PAST_USER'},
                                {'time':2,'question':'PAST_QUESTION','answer':'PAST_ANSWER','options':['A. Red']},
                                {'time':10,'role':'assistant','content':'AT_QUERY'},
                                {'time':11,'role':'assistant','content':'FUTURE'}]
        prompt=build_request(row)['prompt']
        for text in ('PAST_USER','PAST_QUESTION','PAST_ANSWER'):self.assertIn(text,prompt)
        for text in ('AT_QUERY','FUTURE'):self.assertNotIn(text,prompt)
        row['dialogue_history']=row.pop('dialogue_events')
        self.assertEqual(build_request(row)['prompt'],prompt)

    def test_causal_interval(self):
        self.assertIn('0.000s to 9.000s',build_request(self.row(),end=9)['prompt'])
        for end in (-1,11):
            with self.assertRaises(ValueError):build_request(self.row(),end=end)

if __name__=='__main__':unittest.main()
