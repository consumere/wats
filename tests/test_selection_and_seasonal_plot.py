import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd


SOURCE = ast.parse(Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8"))


class SelectionAndSeasonalPlotTest(unittest.TestCase):
    def test_selector_defaults_to_last_two_columns(self):
        self.assertEqual(["third", "fourth"], self.selector_default(["first", "second", "third", "fourth"]))

    def test_selector_defaults_to_only_column_when_one_exists(self):
        self.assertEqual(["only"], self.selector_default(["only"]))

    def selector_default(self, all_columns):
        for node in ast.walk(SOURCE):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "multiselect" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "Select columns to plot"):
                default = next(keyword.value for keyword in node.keywords if keyword.arg == "default")
                return eval(compile(ast.Expression(default), "app.py", "eval"), {"all_columns": all_columns})
        self.fail("Column selector not found")

    def test_seasonal_plot_renders_month_labels_for_one_column(self):
        function = next(node for node in SOURCE.body
                        if isinstance(node, ast.FunctionDef) and node.name == "plot_seasonal_decomposition")
        namespace = {"pd": pd, "plt": plt, "st": SimpleNamespace(warning=lambda message: None)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "app.py", "exec"), namespace)
        data = pd.DataFrame({"rain": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-01", "2020-02-01"]))

        original_boxplot = Axes.boxplot

        def boxplot_without_labels(axis, *args, **kwargs):
            if "labels" in kwargs:
                raise TypeError("Axes.boxplot() got an unexpected keyword argument 'labels'")
            return original_boxplot(axis, *args, **kwargs)

        with patch.object(Axes, "boxplot", boxplot_without_labels):
            figure = namespace["plot_seasonal_decomposition"](data, ["rain"])
        try:
            labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
            self.assertEqual(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], labels)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
