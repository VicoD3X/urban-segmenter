import numpy as np

from src.utils.utils_api import send_image_to_api


def test_send_image_to_api_mock(monkeypatch):
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

    # Mock la réponse API
    def fake_post(url, json):
        class FakeResp:
            status_code = 200
            def json(self):
                return {"mask_pred": [[0, 1], [1, 0]]}
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)

    mask = send_image_to_api(dummy_img, "http://fake")
    assert mask.shape == (2, 2)
