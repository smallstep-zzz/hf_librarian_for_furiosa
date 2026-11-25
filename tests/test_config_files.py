import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hf_librarian import Catalog, Librarian


class ConfigFileTests(unittest.TestCase):
    def catalog(self, root, filename=None):
        model = {'id': 'org/model', 'categories': ['image-to-3d']}
        if filename is not None:
            model['config_file'] = filename
        path = root / 'catalog.json'
        path.write_text(json.dumps({'schema_version': 1, 'settings': {
            'max_concurrent_downloads': 2, 'min_free_gib': .001},
            'categories': {'image-to-3d': '3D'}, 'blacklist': {}, 'models': [model]}))
        return Catalog(path)

    def test_nested_yaml_and_default_json_reuse_only_the_requested_file(self):
        for filename in ('dit/config.yaml', 'vae/config.yml', 'config.json'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                file = root / 'cache/snapshot' / filename
                file.parent.mkdir(parents=True)
                file.write_text('x: 1' if filename.endswith(('.yaml', '.yml')) else '{}')
                info = SimpleNamespace(sha='a'*40, safetensors=None, card_data={'license': 'mit'},
                    siblings=[SimpleNamespace(rfilename=filename, size=file.stat().st_size)])
                with patch('hf_librarian.library.HfApi') as api, \
                        patch('huggingface_hub.hf_hub_download', return_value=str(file)) as download, \
                        patch('hf_librarian.library.snapshot_download', side_effect=AssertionError('weights requested')):
                    api.return_value.model_info.return_value = info
                    with Librarian(self.catalog(root, filename), root/'cache') as store:
                        self.assertEqual(store.request(model_id='org/model', profile='config'), str(file.parent))
                        api.return_value.model_info.side_effect = AssertionError('cache should be offline')
                        self.assertEqual(store.request(model_id='org/model', profile='config'), str(file.parent))
                    download.assert_called_once_with('org/model', filename, revision='a'*40, cache_dir=str((root/'cache').resolve()))

    def test_invalid_config_paths_fail_before_network(self):
        for filename in ('../config.yaml', '/config.json', 'a//b.yaml', 'a/./b.yaml',
                         'a\\b.json', 'weights.safetensors', '*.yaml', 'a\x00.yaml'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with patch('hf_librarian.library.HfApi') as api:
                    with Librarian(self.catalog(root, filename), root/'cache') as store:
                        with self.assertRaises(ValueError):
                            store.request(model_id='org/model', profile='config')
                    api.assert_not_called()

    def test_downloaded_size_is_bounded_even_if_metadata_is_wrong(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root/'cache/config.yaml'
            file.parent.mkdir()
            file.write_bytes(b'x'*(1024**2+1))
            info = SimpleNamespace(sha='a'*40, safetensors=None, card_data={},
                siblings=[SimpleNamespace(rfilename='config.yaml', size=2)])
            with patch('hf_librarian.library.HfApi') as api, \
                    patch('huggingface_hub.hf_hub_download', return_value=str(file)):
                api.return_value.model_info.return_value = info
                with Librarian(self.catalog(root, 'config.yaml'), root/'cache') as store:
                    with self.assertRaises(ValueError):
                        store.request(model_id='org/model', profile='config')


if __name__ == '__main__':
    unittest.main()
