import tempfile,unittest,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
from hf_librarian.overnight import run
class OvernightTests(unittest.TestCase):
    def test_gated_metadata_is_skipped_before_config_or_compute(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'plan.json';plan.write_text(json.dumps({'root':str(root/'batch'),'models':[],'catalog':'unused'}))
            api=MagicMock();api.list_models.return_value=[SimpleNamespace(id='org/restricted',gated=False)]
            api.model_info.return_value=SimpleNamespace(gated='manual')
            with patch('hf_librarian.overnight.HubResearch') as gate, patch('hf_librarian.overnight.Catalog'),patch('hf_librarian.overnight.HfApi',return_value=api),patch('hf_librarian.overnight.hf_hub_download') as download,patch('hf_librarian.overnight.start') as start:
                gate.return_value.call.side_effect=lambda fn: fn()
                run(plan,'2099-01-01T08:00:00')
            download.assert_not_called();start.assert_not_called()
            self.assertTrue((root/'batch-overnight/finished.json').is_file())
if __name__=='__main__':unittest.main()
