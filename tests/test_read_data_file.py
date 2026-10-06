import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def load_reader():
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    nodes = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "TIME_COLUMNS" in names:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"read_data_file", "concatenate_dataframes", "clear_uploaded_data"}:
            nodes.append(node)

    test_module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {"pd": pd, "os": os, "st": SimpleNamespace(session_state={})}
    exec(compile(test_module, "app.py", "exec"), namespace)
    return namespace


reader = load_reader()
read_data_file = reader["read_data_file"]
concatenate_dataframes = reader["concatenate_dataframes"]

WASIM_SAMPLE = """YY\tMM\tDD\tHH\t\"2\"\t\"3\"\t\"4\"\ttot_average
wind_speed interpolated with bicubic spline interpolation
--\t--\t--\t--\t0.1\t0.2\t0.3\t1.0
2010\t1\t1\t12\t5.401\t4.344\t3.407\t4.263
2010\t1\t2\t12\t2.773\t2.043\t1.534\t2.022
"""


class ReadDataFileTest(unittest.TestCase):
    def test_preserves_zero_values_but_masks_nodata(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory, "values.txt")
            fixture.write_text("YY MM DD HH rain\n2000 1 1 24 0\n2000 1 2 24 -9999\n")
            df = read_data_file(fixture)

        self.assertEqual(0, df.iloc[0]["rain"])
        self.assertTrue(pd.isna(df.iloc[1]["rain"]))

    def test_empty_file_reports_format_error(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory, "empty.txt")
            fixture.write_text("")
            with self.assertRaisesRegex(ValueError, "empty.txt is empty"):
                read_data_file(fixture)

    def test_clear_uploaded_data_resets_both_uploads_and_widget_state(self):
        state = reader["st"].session_state
        state.update({"upload_generation": 2, "ts_upload_2": ["series"],
                      "nc_upload_2": ["grid"], "var_select_0": "rain"})

        reader["clear_uploaded_data"]()

        self.assertEqual({"upload_generation": 3}, state)

    def test_reads_wasim_mos_file_with_description_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory, "windsinn100.fut.2010")
            fixture.write_text(WASIM_SAMPLE)
            df = read_data_file(fixture)

        self.assertEqual((2, 4), df.shape)
        self.assertEqual(pd.Timestamp("2010-01-01"), df.index[0])
        self.assertEqual(["2", "3", "4", "tot_average"], list(df.columns))
        self.assertAlmostEqual(5.401, df.loc[pd.Timestamp("2010-01-01"), "2"])
        self.assertIn("wind_speed interpolated", df.attrs["metadata"]["parameter"])

    def test_reads_station_tsv_with_repeated_metadata_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory, "Thulba.tsv")
            fixture.write_text("YY\tMM\tDD\tHH\tOberthulba\tSchlimpfhof\n"
                               "YY\tMM\tDD\tHH\t77.8\t12.8\n"
                               "YY\tMM\tDD\tHH\t568491\t568975\n"
                               "1981\t11\t1\t24\t0.9239692\t0.89775\n"
                               "1981\t11\t2\t24\t0.9994859\t1.11375\n")
            df = read_data_file(fixture)

        self.assertEqual((2, 2), df.shape)
        self.assertEqual(pd.Timestamp("1981-11-01"), df.index[0])
        self.assertEqual(["Oberthulba", "Schlimpfhof"], list(df.columns))
        self.assertAlmostEqual(0.9239692, df.iloc[0]["Oberthulba"])

    def test_rejects_non_timeseries_file_with_clear_error(self):
        fixture = Path(__file__).with_name("invalid_hydrostats.txt")

        with self.assertRaisesRegex(ValueError, "Expected first columns: YY MM DD HH"):
            read_data_file(fixture)

    def test_ignores_wasim_footer_after_timeseries_rows(self):
        fixture = Path(__file__).with_name("footer_timeseries.txt")

        df = read_data_file(fixture)

        self.assertEqual((2, 2), df.shape)
        self.assertEqual(pd.Timestamp("2015-01-02"), df.index[-1])
        self.assertEqual(["13", "14"], list(df.columns))

    def test_mixed_valid_files_can_be_concatenated_after_skipping_invalid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory, "windsinn100.fut.2010")
            fixture.write_text(WASIM_SAMPLE)
            valid = read_data_file(fixture)

        fixture = Path(__file__).with_name("invalid_hydrostats.txt")
        with self.assertRaises(ValueError):
            read_data_file(fixture)

        combined = concatenate_dataframes([valid])

        self.assertEqual(valid.shape, combined.shape)
        self.assertEqual(list(valid.columns), list(combined.columns))


if __name__ == "__main__":
    unittest.main()
