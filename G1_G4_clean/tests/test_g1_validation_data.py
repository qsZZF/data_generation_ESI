import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.prepare_g1_validation_eeg import download_edf


class ValidationDownloadTests(unittest.TestCase):
    def response(self,payload):
        response=MagicMock()
        response.__enter__.return_value=response
        response.headers={'Content-Length':str(len(payload))}
        response.iter_content.return_value=[payload]
        return response

    def test_changed_payload_is_not_promoted_over_recorded_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recording.edf'
            with patch('scripts.prepare_g1_validation_eeg.requests.get',return_value=self.response(b'changed')), \
                 patch('scripts.prepare_g1_validation_eeg.read_edf',return_value=SimpleNamespace(close=lambda:None)):
                with self.assertRaisesRegex(ValueError,'recorded SHA256'):
                    download_edf('https://example.invalid/recording',path,expected_hash=hashlib.sha256(b'original').hexdigest())
            self.assertFalse(path.exists())

    def test_matching_payload_is_atomically_promoted(self):
        payload=b'verified fixture';digest=hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recording.edf'
            with patch('scripts.prepare_g1_validation_eeg.requests.get',return_value=self.response(payload)), \
                 patch('scripts.prepare_g1_validation_eeg.read_edf',return_value=SimpleNamespace(close=lambda:None)):
                self.assertEqual(download_edf('https://example.invalid/recording',path,expected_hash=digest),digest)
            self.assertEqual(path.read_bytes(),payload)
