"""Real tiny-Qwen + decoded synthetic video; full CLI and checkpoint coverage."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import meminstream
from . import test_qwen_stream as native_tests


class DriverTests(unittest.TestCase):
    def test_full_video_causal_query_resume_and_fingerprint(self):
        with tempfile.TemporaryDirectory(prefix='dscache-driver-') as tmp:
            root=Path(tmp);data=root/'data';out=root/'output'
            (data/'statistics').mkdir(parents=True);(data/'videos').mkdir()
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi',
                            '-i','color=c=red:size=56x56:rate=5','-t','2.2','-threads','2',
                            '-pix_fmt','yuv420p',str(data/'videos/test.mp4')],check=True)
            row={'video_path':'./videos/test.mp4','video_id':'tiny','source_dataset':'fixture',
                 'source_id':'tiny1','question':'Which color is visible?', 'query_time':.5,
                 'question_type':'multiple-choice','options':['A. Red','B. Blue'],
                 'correct_index':0,'correct_option':'A','answer':'Red'}
            (data/'annotations.json').write_text(json.dumps([row]))
            (data/'statistics/video_metadata.json').write_text(json.dumps([
                {'file':'test.mp4','width':56,'height':56,'duration':2.2}]))
            def load(config):
                fixture=native_tests.NativeQwenTests('test_strict_parser_and_storage_aliases')
                fixture.setUp();adapter=fixture.adapter;adapter.close_stream()
                # Only the test generation length differs from the real run.
                config.max_new_tokens=2
                adapter.config=config;adapter.policy.config=config
                return adapter
            args=['test','--output-dir',str(out),'--data-root',str(data),
                  '--device','cpu','--budget-gib','0.875']
            with patch.object(sys,'argv',args),patch.object(meminstream.QwenStreamAdapter,'load',side_effect=load):
                meminstream.main()
            pred=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
            done=[json.loads(x) for x in (out/'video_completion.jsonl').read_text().splitlines()]
            self.assertEqual(len(pred),1)
            self.assertEqual(pred[0]['prompt_mode'],'single')
            self.assertEqual(pred[0]['dscache_metrics']['frames_seen'],1,'future frame entered query history')
            self.assertEqual(done[0]['sampled_frames_consumed'],3)
            self.assertEqual(done[0]['decoded_source_frames'],11)
            self.assertTrue(done[0]['decoder_eof'])
            self.assertTrue(done[0]['recent_features_build_kv_on_demand'])
            self.assertEqual(done[0]['dscache_state']['short_pairs'],2)
            self.assertEqual(json.loads((out/'progress.json').read_text())['phase'],'completed')
            manifest_before=(out/'manifest.json').read_bytes()
            annotations_before=(out/'annotations.json').read_bytes()
            with patch.object(sys,'argv',args):
                with self.assertRaises(FileExistsError):meminstream.main()
            self.assertEqual((out/'manifest.json').read_bytes(),manifest_before)
            self.assertEqual((out/'annotations.json').read_bytes(),annotations_before)
            import fcntl
            with (out/'run.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(sys,'argv',args+['--resume']):
                    with self.assertRaises(BlockingIOError):meminstream.main()
            with patch.object(sys,'argv',args+['--resume']),patch.object(meminstream.QwenStreamAdapter,'load',side_effect=load):
                meminstream.main()
            self.assertEqual(len((out/'predictions.jsonl').read_text().splitlines()),1)
            self.assertEqual(len((out/'video_completion.jsonl').read_text().splitlines()),1)
            with patch.object(sys,'argv',args+['--resume','--budget-gib','1.75']):
                with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                    meminstream.main()

if __name__=='__main__':unittest.main()
