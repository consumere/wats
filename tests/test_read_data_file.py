import ast
import unittest
from pathlib import Path

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
        elif isinstance(node, ast.FunctionDef) and node.name in {"read_data_file", "concatenate_dataframes"}:
            nodes.append(node)

    test_module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {"pd": pd}
    exec(compile(test_module, "app.py", "exec"), namespace)
    return namespace["read_data_file"], namespace["concatenate_dataframes"]


read_data_file, concatenate_dataframes = load_reader()


class ReadDataFileTest(unittest.TestCase):
    def test_reads_wasim_mos_file_with_description_and_weights(self):
        df = read_data_file("windsinn100.fut.2010")

        self.assertEqual((32768, 4), df.shape)
        self.assertEqual(pd.Timestamp("2010-01-01"), df.index[0])
        self.assertEqual(["2", "3", "4", "tot_average"], list(df.columns))
        self.assertAlmostEqual(5.401, df.loc[pd.Timestamp("2010-01-01"), "2"])
        self.assertIn("wind_speed interpolated", df.attrs["metadata"]["parameter"])

    def test_reads_station_tsv_with_repeated_metadata_headers(self):
        df = read_data_file("Thulba.tsv")

        self.assertEqual((10288, 2), df.shape)
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
        valid = read_data_file("windsinn100.fut.2010")

        fixture = Path(__file__).with_name("invalid_hydrostats.txt")
        with self.assertRaises(ValueError):
            read_data_file(fixture)

        combined = concatenate_dataframes([valid])

        self.assertEqual(valid.shape, combined.shape)
        self.assertEqual(list(valid.columns), list(combined.columns))


if __name__ == "__main__":
    unittest.main()
