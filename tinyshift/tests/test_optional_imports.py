import unittest
from unittest.mock import patch

from tinyshift.plot.calibration import efficiency_curve


class OptionalImportTests(unittest.TestCase):
    def test_plot_functions_use_the_plot_extra_in_error_message(self):
        with patch(
            "tinyshift.utils.imports.importlib.util.find_spec", return_value=None
        ):
            with self.assertRaisesRegex(ImportError, r"tinyshift\[plot\]"):
                efficiency_curve(clf=None, X=[[0]], y=[0])


if __name__ == "__main__":
    unittest.main()
