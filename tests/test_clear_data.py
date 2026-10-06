import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class ClearDataTest(unittest.TestCase):
    def test_clear_button_resets_upload_generation(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=90).run()

        self.assertFalse(app.exception)
        clear_button = next(button for button in app.button if button.label == "Clear data")
        clear_button.click().run()

        self.assertFalse(app.exception)
        self.assertEqual(1, app.session_state["upload_generation"])


if __name__ == "__main__":
    unittest.main()
