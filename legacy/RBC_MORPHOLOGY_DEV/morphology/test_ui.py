"""Local Streamlit AppTest checks, NOT live independent-profile acceptance."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'morphology_app.py'

class MorphologyUITests(unittest.TestCase):
    def app(self):return AppTest.from_file(str(APP),default_timeout=30).run()
    def analyze_demo(self):
        a=self.app();a.checkbox(key='identity_0').check().run();a.button(key='analyze_0').click().run()
        self.assertFalse(a.exception);self.assertTrue(a.success);return a
    def test_fresh_load(self):
        a=self.app();self.assertFalse(a.exception);self.assertFalse(a.success)
    def test_identity_gate(self):
        a=self.app();a.button(key='analyze_0').click().run();self.assertTrue(a.error);self.assertFalse(a.success)
    def test_analysis_demo(self):self.analyze_demo()
    def test_qc_table_view(self):
        a=self.analyze_demo();a.radio(key='view_0').set_value('All instances / QC').run();self.assertFalse(a.exception);self.assertEqual(len(a.dataframe),1)
    def test_summary_view(self):
        a=self.analyze_demo();a.radio(key='view_0').set_value('Image summaries').run();self.assertFalse(a.exception);self.assertEqual(len(a.dataframe),1)
    def test_download_view(self):
        a=self.analyze_demo();a.radio(key='view_0').set_value('Report / exports').run();self.assertFalse(a.exception);self.assertEqual(len(a.get('download_button')),5)
    def test_setting_change_invalidates_result(self):
        a=self.analyze_demo();a.checkbox(key='exclude_0').check().run();self.assertFalse(a.success);self.assertNotIn('current_result',a.session_state)
    def test_new_input_invalidates_result(self):
        a=self.analyze_demo();a.selectbox(key='fixture_0').select('ellipse').run();self.assertFalse(a.success)
    def test_reset_and_interact(self):
        a=self.analyze_demo();a.button[-1].click().run() if a.button[-1].label=='Reset morphology session' else next(b for b in a.button if b.label=='Reset morphology session').click().run()
        self.assertEqual(a.session_state['generation'],1);self.assertNotIn('current_result',a.session_state)
        self.assertFalse(a.checkbox(key='identity_1').value)
        a.checkbox(key='exclude_1').check().run();self.assertFalse(a.success)
        a.checkbox(key='identity_1').check().run();a.button(key='analyze_1').click().run();self.assertTrue(a.success)
    def test_missing_calibration_source(self):
        a=self.app();a.checkbox(key='calibrated_0').check().run();a.checkbox(key='identity_0').check().run();a.button(key='analyze_0').click().run();self.assertTrue(a.error)
    def test_design_only_pages(self):
        a=self.app()
        for name in a.radio(key='module_0').options[1:]:
            a.radio(key='module_0').set_value(name).run();self.assertFalse(a.exception)
            self.assertEqual(len(a.get('file_uploader')),0)
    def test_component_label(self):
        a=self.app();a.selectbox(key='kind_0').select('Binary connected components — NOT identified cells').run()
        a.button(key='analyze_0').click().run();self.assertTrue(a.success)
        self.assertIn('unestablished',a.session_state['current_result']['instances'][0]['object_semantics'])

if __name__=='__main__':unittest.main()
