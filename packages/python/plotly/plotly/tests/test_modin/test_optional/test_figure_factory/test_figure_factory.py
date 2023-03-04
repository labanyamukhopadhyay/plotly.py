from unittest import TestCase
from plotly import optional_imports
from plotly.graph_objs import graph_objs as go
from plotly.exceptions import PlotlyError
import plotly.io as pio

import plotly.figure_factory as ff
from plotly.tests.test_optional.optional_utils import NumpyTestUtilsMixin

import numpy as np
from plotly.tests.utils import TestCaseNoTemplate
from scipy.spatial import Delaunay
import modin.pandas as pd

shapely = optional_imports.get_module("shapely")
shapefile = optional_imports.get_module("shapefile")
gp = optional_imports.get_module("geopandas")
sk_measure = optional_imports.get_module("skimage")


class TestScatterPlotMatrix(NumpyTestUtilsMixin, TestCaseNoTemplate):
    def test_one_column_dataframe(self):

        # check: dataframe has 1 column or less
        df = pd.DataFrame([1, 2, 3])

        pattern = (
            "Dataframe has only one column. To use the scatterplot matrix, "
            "use at least 2 columns."
        )

        self.assertRaisesRegex(PlotlyError, pattern, ff.create_scatterplotmatrix, df)

    def test_valid_diag_choice(self):

        # make sure that the diagonal param is valid
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

        self.assertRaises(PlotlyError, ff.create_scatterplotmatrix, df, diag="foo")

    def test_forbidden_params(self):

        # check: the forbidden params of 'marker' in **kwargs
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

        kwargs = {"marker": {"size": 15}}

        pattern = (
            "Your kwargs dictionary cannot include the 'size', 'color' or "
            "'colorscale' key words inside the marker dict since 'size' is "
            "already an argument of the scatterplot matrix function and both "
            "'color' and 'colorscale are set internally."
        )

        self.assertRaisesRegex(
            PlotlyError, pattern, ff.create_scatterplotmatrix, df, **kwargs
        )

    def test_valid_index_choice(self):

        # check: index is a column name
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["apple", "pear"])

        pattern = (
            "Make sure you set the index input variable to one of the column "
            "names of your dataframe."
        )

        self.assertRaisesRegex(
            PlotlyError, pattern, ff.create_scatterplotmatrix, df, index="grape"
        )

    def test_same_data_in_dataframe_columns(self):

        # check: either all numbers or strings in each dataframe column
        df = pd.DataFrame([["a", 2], [3, 4]])

        pattern = (
            "Error in dataframe. Make sure all entries of each column are "
            "either numbers or strings."
        )

        self.assertRaisesRegex(PlotlyError, pattern, ff.create_scatterplotmatrix, df)

        df = pd.DataFrame([[1, 2], ["a", 4]])

        self.assertRaisesRegex(PlotlyError, pattern, ff.create_scatterplotmatrix, df)

    def test_same_data_in_index(self):

        # check: either all numbers or strings in index column
        df = pd.DataFrame([["a", 2], [3, 4]], columns=["apple", "pear"])

        pattern = (
            "Error in indexing column. Make sure all entries of each column "
            "are all numbers or all strings."
        )

        self.assertRaisesRegex(
            PlotlyError, pattern, ff.create_scatterplotmatrix, df, index="apple"
        )

        df = pd.DataFrame([[1, 2], ["a", 4]], columns=["apple", "pear"])

        self.assertRaisesRegex(
            PlotlyError, pattern, ff.create_scatterplotmatrix, df, index="apple"
        )

    def test_valid_colormap(self):

        # check: the colormap argument is in a valid form
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])

        # check: valid plotly scalename is entered
        self.assertRaises(
            PlotlyError,
            ff.create_scatterplotmatrix,
            df,
            index="a",
            colormap="fake_scale",
        )

        pattern_rgb = (
            "Whoops! The elements in your rgb colors tuples cannot " "exceed 255.0."
        )

        # check: proper 'rgb' color
        self.assertRaisesRegex(
            PlotlyError,
            pattern_rgb,
            ff.create_scatterplotmatrix,
            df,
            colormap="rgb(500, 1, 1)",
            index="c",
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern_rgb,
            ff.create_scatterplotmatrix,
            df,
            colormap=["rgb(500, 1, 1)"],
            index="c",
        )

        pattern_tuple = (
            "Whoops! The elements in your colors tuples cannot " "exceed 1.0."
        )

        # check: proper color tuple
        self.assertRaisesRegex(
            PlotlyError,
            pattern_tuple,
            ff.create_scatterplotmatrix,
            df,
            colormap=(2, 1, 1),
            index="c",
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern_tuple,
            ff.create_scatterplotmatrix,
            df,
            colormap=[(2, 1, 1)],
            index="c",
        )

    def test_valid_endpts(self):

        # check: the endpts is a list or a tuple
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])

        pattern = (
            "The intervals_endpts argument must be a list or tuple of a "
            "sequence of increasing numbers."
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_scatterplotmatrix,
            df,
            index="a",
            colormap="Hot",
            endpts="foo",
        )

        # check: the endpts are a list of numbers
        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_scatterplotmatrix,
            df,
            index="a",
            colormap="Hot",
            endpts=["a"],
        )

        # check: endpts is a list of INCREASING numbers
        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_scatterplotmatrix,
            df,
            index="a",
            colormap="Hot",
            endpts=[2, 1],
        )

    def test_dictionary_colormap(self):

        # if colormap is a dictionary, make sure it all the values in the
        # index column are keys in colormap
        df = pd.DataFrame(
            [["apple", "happy"], ["pear", "sad"]], columns=["Fruit", "Emotion"]
        )

        colormap = {"happy": "rgb(5, 5, 5)"}

        pattern = (
            "If colormap is a dictionary, all the names in the index " "must be keys."
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_scatterplotmatrix,
            df,
            index="Emotion",
            colormap=colormap,
        )

    def test_scatter_plot_matrix(self):

        # check if test scatter plot matrix without index or theme matches
        # with the expected output
        df = pd.DataFrame(
            [
                [2, "Apple"],
                [6, "Pear"],
                [-15, "Apple"],
                [5, "Pear"],
                [-2, "Apple"],
                [0, "Apple"],
            ],
            columns=["Numbers", "Fruit"],
        )

        test_scatter_plot_matrix = ff.create_scatterplotmatrix(
            df=df,
            diag="box",
            height=1000,
            width=1000,
            size=13,
            title="Scatterplot Matrix",
        )

        exp_scatter_plot_matrix = {
            "data": [
                {
                    "showlegend": False,
                    "type": "box",
                    "xaxis": "x",
                    "y": [2, 6, -15, 5, -2, 0],
                    "yaxis": "y",
                },
                {
                    "marker": {"size": 13},
                    "mode": "markers",
                    "showlegend": False,
                    "type": "scatter",
                    "x": ["Apple", "Pear", "Apple", "Pear", "Apple", "Apple"],
                    "xaxis": "x2",
                    "y": [2, 6, -15, 5, -2, 0],
                    "yaxis": "y2",
                },
                {
                    "marker": {"size": 13},
                    "mode": "markers",
                    "showlegend": False,
                    "type": "scatter",
                    "x": [2, 6, -15, 5, -2, 0],
                    "xaxis": "x3",
                    "y": ["Apple", "Pear", "Apple", "Pear", "Apple", "Apple"],
                    "yaxis": "y3",
                },
                {
                    "name": None,
                    "showlegend": False,
                    "type": "box",
                    "xaxis": "x4",
                    "y": ["Apple", "Pear", "Apple", "Pear", "Apple", "Apple"],
                    "yaxis": "y4",
                },
            ],
            "layout": {
                "height": 1000,
                "showlegend": True,
                "title": {"text": "Scatterplot Matrix"},
                "width": 1000,
                "xaxis": {
                    "anchor": "y",
                    "domain": [0.0, 0.45],
                    "showticklabels": False,
                },
                "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0]},
                "xaxis3": {
                    "anchor": "y3",
                    "domain": [0.0, 0.45],
                    "title": {"text": "Numbers"},
                },
                "xaxis4": {
                    "anchor": "y4",
                    "domain": [0.55, 1.0],
                    "showticklabels": False,
                    "title": {"text": "Fruit"},
                },
                "yaxis": {
                    "anchor": "x",
                    "domain": [0.575, 1.0],
                    "title": {"text": "Numbers"},
                },
                "yaxis2": {"anchor": "x2", "domain": [0.575, 1.0]},
                "yaxis3": {
                    "anchor": "x3",
                    "domain": [0.0, 0.425],
                    "title": {"text": "Fruit"},
                },
                "yaxis4": {"anchor": "x4", "domain": [0.0, 0.425]},
            },
        }

        self.assert_fig_equal(
            test_scatter_plot_matrix["data"][0], exp_scatter_plot_matrix["data"][0]
        )

        self.assert_fig_equal(
            test_scatter_plot_matrix["data"][1], exp_scatter_plot_matrix["data"][1]
        )

        self.assert_fig_equal(
            test_scatter_plot_matrix["layout"], exp_scatter_plot_matrix["layout"]
        )

    def test_scatter_plot_matrix_kwargs(self):

        # check if test scatter plot matrix matches with
        # the expected output
        df = pd.DataFrame(
            [
                [2, "Apple"],
                [6, "Pear"],
                [-15, "Apple"],
                [5, "Pear"],
                [-2, "Apple"],
                [0, "Apple"],
            ],
            columns=["Numbers", "Fruit"],
        )

        test_scatter_plot_matrix = ff.create_scatterplotmatrix(
            df,
            index="Fruit",
            endpts=[-10, -1],
            diag="histogram",
            height=1000,
            width=1000,
            size=13,
            title="Scatterplot Matrix",
            colormap="YlOrRd",
            marker=dict(symbol=136),
        )

        exp_scatter_plot_matrix = {
            "data": [
                {
                    "marker": {"color": "rgb(128, 0, 38)"},
                    "showlegend": False,
                    "type": "histogram",
                    "x": [2, -15, -2, 0],
                    "xaxis": "x",
                    "yaxis": "y",
                },
                {
                    "marker": {"color": "rgb(255, 255, 204)"},
                    "showlegend": False,
                    "type": "histogram",
                    "x": [6, 5],
                    "xaxis": "x",
                    "yaxis": "y",
                },
            ],
            "layout": {
                "barmode": "stack",
                "height": 1000,
                "showlegend": True,
                "title": {"text": "Scatterplot Matrix"},
                "width": 1000,
                "xaxis": {
                    "anchor": "y",
                    "domain": [0.0, 1.0],
                    "title": {"text": "Numbers"},
                },
                "yaxis": {
                    "anchor": "x",
                    "domain": [0.0, 1.0],
                    "title": {"text": "Numbers"},
                },
            },
        }

        self.assert_fig_equal(
            test_scatter_plot_matrix["data"][0], exp_scatter_plot_matrix["data"][0]
        )

        self.assert_fig_equal(
            test_scatter_plot_matrix["data"][1], exp_scatter_plot_matrix["data"][1]
        )

        self.assert_fig_equal(
            test_scatter_plot_matrix["layout"], exp_scatter_plot_matrix["layout"]
        )


class TestGantt(NumpyTestUtilsMixin, TestCaseNoTemplate):
    def test_df_dataframe(self):

        # validate dataframe has correct column names
        df1 = pd.DataFrame([[2, "Apple"]], columns=["Numbers", "Fruit"])
        self.assertRaises(PlotlyError, ff.create_gantt, df1)

    def test_df_dataframe_all_args(self):

        # check if gantt chart matches with expected output

        df = pd.DataFrame(
            [
                ["Job A", "2009-01-01", "2009-02-30"],
                ["Job B", "2009-03-05", "2009-04-15"],
            ],
            columns=["Task", "Start", "Finish"],
        )

        test_gantt_chart = ff.create_gantt(df)

        exp_gantt_chart = go.Figure(
            **{
                "data": [
                    {
                        "x": ("2009-03-05", "2009-04-15", "2009-04-15", "2009-03-05"),
                        "y": [0.8, 0.8, 1.2, 1.2],
                        "mode": "none",
                        "fill": "toself",
                        "hoverinfo": "name",
                        "fillcolor": "rgb(255, 127, 14)",
                        "name": "Job B",
                        "legendgroup": "rgb(255, 127, 14)",
                    },
                    {
                        "x": ("2009-01-01", "2009-02-30", "2009-02-30", "2009-01-01"),
                        "y": [-0.2, -0.2, 0.2, 0.2],
                        "mode": "none",
                        "fill": "toself",
                        "hoverinfo": "name",
                        "fillcolor": "rgb(31, 119, 180)",
                        "name": "Job A",
                        "legendgroup": "rgb(31, 119, 180)",
                    },
                    {
                        "x": ("2009-03-05", "2009-04-15"),
                        "y": [1, 1],
                        "mode": "markers",
                        "text": [None, None],
                        "marker": {
                            "color": "rgb(255, 127, 14)",
                            "size": 1,
                            "opacity": 0,
                        },
                        "name": "",
                        "showlegend": False,
                        "legendgroup": "rgb(255, 127, 14)",
                    },
                    {
                        "x": ("2009-01-01", "2009-02-30"),
                        "y": [0, 0],
                        "mode": "markers",
                        "text": [None, None],
                        "marker": {
                            "color": "rgb(31, 119, 180)",
                            "size": 1,
                            "opacity": 0,
                        },
                        "name": "",
                        "showlegend": False,
                        "legendgroup": "rgb(31, 119, 180)",
                    },
                ],
                "layout": {
                    "title": "Gantt Chart",
                    "showlegend": False,
                    "height": 600,
                    "width": 900,
                    "shapes": [],
                    "hovermode": "closest",
                    "yaxis": {
                        "showgrid": False,
                        "ticktext": ["Job A", "Job B"],
                        "tickvals": [0, 1],
                        "range": [-1, 3],
                        "autorange": False,
                        "zeroline": False,
                    },
                    "xaxis": {
                        "showgrid": False,
                        "zeroline": False,
                        "rangeselector": {
                            "buttons": [
                                {
                                    "count": 7,
                                    "label": "1w",
                                    "step": "day",
                                    "stepmode": "backward",
                                },
                                {
                                    "count": 1,
                                    "label": "1m",
                                    "step": "month",
                                    "stepmode": "backward",
                                },
                                {
                                    "count": 6,
                                    "label": "6m",
                                    "step": "month",
                                    "stepmode": "backward",
                                },
                                {
                                    "count": 1,
                                    "label": "YTD",
                                    "step": "year",
                                    "stepmode": "todate",
                                },
                                {
                                    "count": 1,
                                    "label": "1y",
                                    "step": "year",
                                    "stepmode": "backward",
                                },
                                {"step": "all"},
                            ]
                        },
                        "type": "date",
                    },
                },
            }
        )

        self.assert_fig_equal(test_gantt_chart["data"][1], exp_gantt_chart["data"][1])
        self.assert_fig_equal(test_gantt_chart["data"][1], exp_gantt_chart["data"][1])
        self.assert_fig_equal(test_gantt_chart["data"][2], exp_gantt_chart["data"][2])
        self.assert_fig_equal(test_gantt_chart["data"][3], exp_gantt_chart["data"][3])


class TestViolin(NumpyTestUtilsMixin, TestCaseNoTemplate):
    def test_data_header(self):

        # make sure data_header is entered

        data = pd.DataFrame([["apple", 2], ["pear", 4]], columns=["a", "b"])

        pattern = (
            "data_header must be the column name with the desired "
            "numeric data for the violin plot."
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_violin,
            data,
            group_header="a",
            colors=["rgb(1, 2, 3)"],
        )

    def test_colors_dict(self):

        # check: if colorscale is True, make sure colors is not a dictionary

        data = pd.DataFrame([["apple", 2], ["pear", 4]], columns=["a", "b"])

        pattern = (
            "The colors param cannot be a dictionary if you are " "using a colorscale."
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_violin,
            data,
            data_header="b",
            group_header="a",
            use_colorscale=True,
            colors={"a": "rgb(1, 2, 3)"},
        )

        # check: colors contains all group names as keys

        pattern2 = (
            "If colors is a dictionary, all the group names must "
            "appear as keys in colors."
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern2,
            ff.create_violin,
            data,
            data_header="b",
            group_header="a",
            use_colorscale=False,
            colors={"a": "rgb(1, 2, 3)"},
        )

    def test_valid_colorscale(self):

        # check: if colorscale is enabled, colors is a list with 2+ items

        data = pd.DataFrame([["apple", 2], ["pear", 4]], columns=["a", "b"])

        pattern = (
            "colors must be a list with at least 2 colors. A Plotly "
            "scale is allowed."
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_violin,
            data,
            data_header="b",
            group_header="a",
            use_colorscale=True,
            colors="rgb(1, 2, 3)",
        )

    def test_group_stats(self):

        # check: group_stats is a dictionary

        data = pd.DataFrame([["apple", 2], ["pear", 4]], columns=["a", "b"])

        pattern = "Your group_stats param must be a dictionary."

        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_violin,
            data,
            data_header="b",
            group_header="a",
            use_colorscale=True,
            colors=["rgb(1, 2, 3)", "rgb(4, 5, 6)"],
            group_stats=1,
        )

        # check: all groups are represented as keys in group_stats

        pattern2 = (
            "All values/groups in the index column must be "
            "represented as a key in group_stats."
        )

        self.assertRaisesRegex(
            PlotlyError,
            pattern2,
            ff.create_violin,
            data,
            data_header="b",
            group_header="a",
            use_colorscale=True,
            colors=["rgb(1, 2, 3)", "rgb(4, 5, 6)"],
            group_stats={"apple": 1},
        )


class TestFacetGrid(NumpyTestUtilsMixin, TestCaseNoTemplate):
    def test_x_and_y_for_scatter(self):
        data = pd.DataFrame([[0, 0], [1, 1]], columns=["a", "b"])

        pattern = (
            "You need to input 'x' and 'y' if you are you are using a "
            "trace_type of 'scatter' or 'scattergl'."
        )

        self.assertRaisesRegex(PlotlyError, pattern, ff.create_facet_grid, data, "a")

    def test_valid_col_selection(self):
        data = pd.DataFrame([[0, 0], [1, 1]], columns=["a", "b"])

        pattern = (
            "x, y, facet_row, facet_col and color_name must be keys in your "
            "dataframe."
        )

        self.assertRaisesRegex(
            PlotlyError, pattern, ff.create_facet_grid, data, "a", "c"
        )

    def test_valid_trace_type(self):
        data = pd.DataFrame([[0, 0], [1, 1]], columns=["a", "b"])

        self.assertRaises(
            PlotlyError, ff.create_facet_grid, data, "a", "b", trace_type="foo"
        )

    def test_valid_scales(self):
        data = pd.DataFrame([[0, 0], [1, 1]], columns=["a", "b"])

        pattern = "'scales' must be set to 'fixed', 'free_x', 'free_y' and 'free'."

        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_facet_grid,
            data,
            "a",
            "b",
            scales="not_free",
        )

    def test_valid_plotly_color_scale_name(self):
        data = pd.DataFrame([[0, 0], [1, 1]], columns=["a", "b"])

        self.assertRaises(
            PlotlyError,
            ff.create_facet_grid,
            data,
            "a",
            "b",
            color_name="a",
            colormap="wrong one",
        )

    def test_facet_labels(self):
        data = pd.DataFrame([["a1", 0], ["a2", 1]], columns=["a", "b"])

        self.assertRaises(
            PlotlyError,
            ff.create_facet_grid,
            data,
            "a",
            "b",
            facet_row="a",
            facet_row_labels={},
        )

        self.assertRaises(
            PlotlyError,
            ff.create_facet_grid,
            data,
            "a",
            "b",
            facet_col="a",
            facet_col_labels={},
        )

    def test_valid_color_dict(self):
        data = pd.DataFrame([[0, 0, "foo"], [1, 1, "foo"]], columns=["a", "b", "foo"])

        pattern = (
            "If using 'colormap' as a dictionary, make sure "
            "all the values of the colormap column are in "
            "the keys of your dictionary."
        )

        color_dict = {"bar": "#ffffff"}

        self.assertRaisesRegex(
            PlotlyError,
            pattern,
            ff.create_facet_grid,
            data,
            "a",
            "b",
            color_name="a",
            colormap=color_dict,
        )

    def test_valid_colorscale_name(self):
        data = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=["a", "b", "c"])

        colormap = "foo"

        self.assertRaises(
            PlotlyError,
            ff.create_facet_grid,
            data,
            "a",
            "b",
            color_name="c",
            colormap=colormap,
        )

    def test_valid_facet_grid_fig(self):
        mpg = [
            ["audi", "a4", 1.8, 1999, 4, "auto(15)", "f", 18, 29, "p", "compact"],
            ["audi", "a4", 1.8, 1999, 4, "auto(l5)", "f", 18, 29, "p", "compact"],
            ["audi", "a4", 2, 2008, 4, "manual(m6)", "f", 20, 31, "p", "compact"],
            ["audi", "a4", 2, 2008, 4, "auto(av)", "f", 21, 30, "p", "compact"],
            ["audi", "a4", 2.8, 1999, 6, "auto(l5)", "f", 16, 26, "p", "compact"],
            ["audi", "a4", 2.8, 1999, 6, "manual(m5)", "f", 18, 26, "p", "compact"],
            ["audi", "a4", 3.1, 2008, 6, "auto(av)", "f", 18, 27, "p", "compact"],
            [
                "audi",
                "a4 quattro",
                1.8,
                1999,
                4,
                "manual(m5)",
                "4",
                18,
                26,
                "p",
                "compact",
            ],
            [
                "audi",
                "a4 quattro",
                1.8,
                1999,
                4,
                "auto(l5)",
                "4",
                16,
                25,
                "p",
                "compact",
            ],
            [
                "audi",
                "a4 quattro",
                2,
                2008,
                4,
                "manual(m6)",
                "4",
                20,
                28,
                "p",
                "compact",
            ],
        ]

        df = pd.DataFrame(
            mpg,
            columns=[
                "manufacturer",
                "model",
                "displ",
                "year",
                "cyl",
                "trans",
                "drv",
                "cty",
                "hwy",
                "fl",
                "class",
            ],
        )
        test_facet_grid = ff.create_facet_grid(df, x="displ", y="cty", facet_col="cyl")

        exp_facet_grid = {
            "data": [
                {
                    "marker": {
                        "color": "rgb(31, 119, 180)",
                        "line": {"color": "darkgrey", "width": 1},
                        "size": 8,
                    },
                    "mode": "markers",
                    "opacity": 0.6,
                    "type": "scatter",
                    "x": [1.8, 1.8, 2.0, 2.0, 1.8, 1.8, 2.0],
                    "xaxis": "x",
                    "y": [18, 18, 20, 21, 18, 16, 20],
                    "yaxis": "y",
                },
                {
                    "marker": {
                        "color": "rgb(31, 119, 180)",
                        "line": {"color": "darkgrey", "width": 1},
                        "size": 8,
                    },
                    "mode": "markers",
                    "opacity": 0.6,
                    "type": "scatter",
                    "x": [2.8, 2.8, 3.1],
                    "xaxis": "x2",
                    "y": [16, 18, 18],
                    "yaxis": "y2",
                },
            ],
            "layout": {
                "annotations": [
                    {
                        "font": {"color": "#0f0f0f", "size": 13},
                        "showarrow": False,
                        "text": "4",
                        "textangle": 0,
                        "x": 0.24625,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": 1.03,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                    {
                        "font": {"color": "#0f0f0f", "size": 13},
                        "showarrow": False,
                        "text": "6",
                        "textangle": 0,
                        "x": 0.7537499999999999,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": 1.03,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                    {
                        "font": {"color": "#000000", "size": 12},
                        "showarrow": False,
                        "text": "displ",
                        "textangle": 0,
                        "x": 0.5,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": -0.1,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                    {
                        "font": {"color": "#000000", "size": 12},
                        "showarrow": False,
                        "text": "cty",
                        "textangle": -90,
                        "x": -0.1,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": 0.5,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                ],
                "height": 600,
                "legend": {
                    "bgcolor": "#efefef",
                    "borderwidth": 1,
                    "x": 1.05,
                    "y": 1,
                    "yanchor": "top",
                },
                "paper_bgcolor": "rgb(251, 251, 251)",
                "showlegend": False,
                "title": {"text": ""},
                "width": 600,
                "xaxis": {
                    "anchor": "y",
                    "domain": [0.0, 0.4925],
                    "dtick": 0,
                    "range": [0.85, 4.1575],
                    "ticklen": 0,
                    "zeroline": False,
                },
                "xaxis2": {
                    "anchor": "y2",
                    "domain": [0.5075, 1.0],
                    "dtick": 0,
                    "range": [0.85, 4.1575],
                    "ticklen": 0,
                    "zeroline": False,
                },
                "yaxis": {
                    "anchor": "x",
                    "domain": [0.0, 1.0],
                    "dtick": 1,
                    "range": [15.75, 21.2625],
                    "ticklen": 0,
                    "zeroline": False,
                },
                "yaxis2": {
                    "anchor": "x2",
                    "domain": [0.0, 1.0],
                    "dtick": 1,
                    "matches": "y",
                    "range": [15.75, 21.2625],
                    "showticklabels": False,
                    "ticklen": 0,
                    "zeroline": False,
                },
            },
        }

        for j in [0, 1]:
            self.assert_fig_equal(test_facet_grid["data"][j], exp_facet_grid["data"][j])

        self.assert_fig_equal(test_facet_grid["layout"], exp_facet_grid["layout"])


class TestBullet(NumpyTestUtilsMixin, TestCaseNoTemplate):
    def test_full_bullet(self):
        data = [
            {
                "title": "Revenue",
                "subtitle": "US$, in thousands",
                "ranges": [150, 225, 300],
                "measures": [220, 270],
                "markers": [250],
            },
            {
                "title": "Profit",
                "subtitle": "%",
                "ranges": [20, 25, 30],
                "measures": [21, 23],
                "markers": [26],
            },
            {
                "title": "Order Size",
                "subtitle": "US$, average",
                "ranges": [350, 500, 600],
                "measures": [100, 320],
                "markers": [550],
            },
            {
                "title": "New Customers",
                "subtitle": "count",
                "ranges": [1400, 2000, 2500],
                "measures": [1000, 1650],
                "markers": [2100],
            },
            {
                "title": "Satisfaction",
                "subtitle": "out of 5",
                "ranges": [3.5, 4.25, 5],
                "measures": [3.2, 4.7],
                "markers": [4.4],
            },
        ]

        df = pd.DataFrame(data)

        measure_colors = ["rgb(255, 127, 14)", "rgb(44, 160, 44)"]
        range_colors = ["rgb(255, 127, 14)", "rgb(44, 160, 44)"]

        fig = ff.create_bullet(
            df,
            orientation="v",
            markers="markers",
            measures="measures",
            ranges="ranges",
            subtitles="subtitle",
            titles="title",
            range_colors=range_colors,
            measure_colors=measure_colors,
            title="new title",
            scatter_options={"marker": {"size": 30, "symbol": "hourglass"}},
        )

        exp_fig = {
            "data": [
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x",
                    "y": [300],
                    "yaxis": "y",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(149.5, 143.5, 29.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x",
                    "y": [225],
                    "yaxis": "y",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x",
                    "y": [150],
                    "yaxis": "y",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x",
                    "y": [270],
                    "yaxis": "y",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x",
                    "y": [220],
                    "yaxis": "y",
                },
                {
                    "hoverinfo": "y",
                    "marker": {
                        "color": "rgb(0, 0, 0)",
                        "size": 30,
                        "symbol": "hourglass",
                    },
                    "name": "markers",
                    "type": "scatter",
                    "x": [0.5],
                    "xaxis": "x",
                    "y": [250],
                    "yaxis": "y",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x2",
                    "y": [30],
                    "yaxis": "y2",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(149.5, 143.5, 29.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x2",
                    "y": [25],
                    "yaxis": "y2",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x2",
                    "y": [20],
                    "yaxis": "y2",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x2",
                    "y": [23],
                    "yaxis": "y2",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x2",
                    "y": [21],
                    "yaxis": "y2",
                },
                {
                    "hoverinfo": "y",
                    "marker": {
                        "color": "rgb(0, 0, 0)",
                        "size": 30,
                        "symbol": "hourglass",
                    },
                    "name": "markers",
                    "type": "scatter",
                    "x": [0.5],
                    "xaxis": "x2",
                    "y": [26],
                    "yaxis": "y2",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x3",
                    "y": [600],
                    "yaxis": "y3",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(149.5, 143.5, 29.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x3",
                    "y": [500],
                    "yaxis": "y3",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x3",
                    "y": [350],
                    "yaxis": "y3",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x3",
                    "y": [320],
                    "yaxis": "y3",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x3",
                    "y": [100],
                    "yaxis": "y3",
                },
                {
                    "hoverinfo": "y",
                    "marker": {
                        "color": "rgb(0, 0, 0)",
                        "size": 30,
                        "symbol": "hourglass",
                    },
                    "name": "markers",
                    "type": "scatter",
                    "x": [0.5],
                    "xaxis": "x3",
                    "y": [550],
                    "yaxis": "y3",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x4",
                    "y": [2500],
                    "yaxis": "y4",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(149.5, 143.5, 29.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x4",
                    "y": [2000],
                    "yaxis": "y4",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x4",
                    "y": [1400],
                    "yaxis": "y4",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x4",
                    "y": [1650],
                    "yaxis": "y4",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x4",
                    "y": [1000],
                    "yaxis": "y4",
                },
                {
                    "hoverinfo": "y",
                    "marker": {
                        "color": "rgb(0, 0, 0)",
                        "size": 30,
                        "symbol": "hourglass",
                    },
                    "name": "markers",
                    "type": "scatter",
                    "x": [0.5],
                    "xaxis": "x4",
                    "y": [2100],
                    "yaxis": "y4",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x5",
                    "y": [5],
                    "yaxis": "y5",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(149.5, 143.5, 29.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x5",
                    "y": [4.25],
                    "yaxis": "y5",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "ranges",
                    "orientation": "v",
                    "type": "bar",
                    "width": 2,
                    "x": [0],
                    "xaxis": "x5",
                    "y": [3.5],
                    "yaxis": "y5",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(44.0, 160.0, 44.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x5",
                    "y": [4.7],
                    "yaxis": "y5",
                },
                {
                    "base": 0,
                    "hoverinfo": "y",
                    "marker": {"color": "rgb(255.0, 127.0, 14.0)"},
                    "name": "measures",
                    "orientation": "v",
                    "type": "bar",
                    "width": 0.4,
                    "x": [0.5],
                    "xaxis": "x5",
                    "y": [3.2],
                    "yaxis": "y5",
                },
                {
                    "hoverinfo": "y",
                    "marker": {
                        "color": "rgb(0, 0, 0)",
                        "size": 30,
                        "symbol": "hourglass",
                    },
                    "name": "markers",
                    "type": "scatter",
                    "x": [0.5],
                    "xaxis": "x5",
                    "y": [4.4],
                    "yaxis": "y5",
                },
            ],
            "layout": {
                "annotations": [
                    {
                        "font": {"color": "#0f0f0f", "size": 13},
                        "showarrow": False,
                        "text": "<b>Revenue</b>",
                        "textangle": 0,
                        "x": 0.019999999999999997,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": 1.03,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                    {
                        "font": {"color": "#0f0f0f", "size": 13},
                        "showarrow": False,
                        "text": "<b>Profit</b>",
                        "textangle": 0,
                        "x": 0.26,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": 1.03,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                    {
                        "font": {"color": "#0f0f0f", "size": 13},
                        "showarrow": False,
                        "text": "<b>Order Size</b>",
                        "textangle": 0,
                        "x": 0.5,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": 1.03,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                    {
                        "font": {"color": "#0f0f0f", "size": 13},
                        "showarrow": False,
                        "text": "<b>New Customers</b>",
                        "textangle": 0,
                        "x": 0.74,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": 1.03,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                    {
                        "font": {"color": "#0f0f0f", "size": 13},
                        "showarrow": False,
                        "text": "<b>Satisfaction</b>",
                        "textangle": 0,
                        "x": 0.98,
                        "xanchor": "center",
                        "xref": "paper",
                        "y": 1.03,
                        "yanchor": "middle",
                        "yref": "paper",
                    },
                ],
                "barmode": "stack",
                "height": 600,
                "margin": {"l": 80},
                "shapes": [],
                "showlegend": False,
                "title": "new title",
                "width": 1000,
                "xaxis1": {
                    "anchor": "y",
                    "domain": [0.0, 0.039999999999999994],
                    "range": [0, 1],
                    "showgrid": False,
                    "showticklabels": False,
                    "zeroline": False,
                },
                "xaxis2": {
                    "anchor": "y2",
                    "domain": [0.24, 0.27999999999999997],
                    "range": [0, 1],
                    "showgrid": False,
                    "showticklabels": False,
                    "zeroline": False,
                },
                "xaxis3": {
                    "anchor": "y3",
                    "domain": [0.48, 0.52],
                    "range": [0, 1],
                    "showgrid": False,
                    "showticklabels": False,
                    "zeroline": False,
                },
                "xaxis4": {
                    "anchor": "y4",
                    "domain": [0.72, 0.76],
                    "range": [0, 1],
                    "showgrid": False,
                    "showticklabels": False,
                    "zeroline": False,
                },
                "xaxis5": {
                    "anchor": "y5",
                    "domain": [0.96, 1.0],
                    "range": [0, 1],
                    "showgrid": False,
                    "showticklabels": False,
                    "zeroline": False,
                },
                "yaxis1": {
                    "anchor": "x",
                    "domain": [0.0, 1.0],
                    "showgrid": False,
                    "tickwidth": 1,
                    "zeroline": False,
                },
                "yaxis2": {
                    "anchor": "x2",
                    "domain": [0.0, 1.0],
                    "showgrid": False,
                    "tickwidth": 1,
                    "zeroline": False,
                },
                "yaxis3": {
                    "anchor": "x3",
                    "domain": [0.0, 1.0],
                    "showgrid": False,
                    "tickwidth": 1,
                    "zeroline": False,
                },
                "yaxis4": {
                    "anchor": "x4",
                    "domain": [0.0, 1.0],
                    "showgrid": False,
                    "tickwidth": 1,
                    "zeroline": False,
                },
                "yaxis5": {
                    "anchor": "x5",
                    "domain": [0.0, 1.0],
                    "showgrid": False,
                    "tickwidth": 1,
                    "zeroline": False,
                },
            },
        }

        for i in range(len(fig["data"])):
            self.assert_fig_equal(fig["data"][i], exp_fig["data"][i])


# class TestChoropleth(NumpyTestUtilsMixin, TestCaseNoTemplate):

# run tests if required packages are installed
# if shapely and shapefile and gp:


# class TestQuiver(TestCaseNoTemplate):
#     def test_scaleratio_param(self):
#         x, y = np.meshgrid(np.arange(0.5, 3.5, 0.5), np.arange(0.5, 4.5, 0.5))
#         u = x
#         v = y
#         angle = np.arctan(v / u)
#         norm = 0.25
#         u = norm * np.cos(angle)
#         v = norm * np.sin(angle)
#         fig = ff.create_quiver(x, y, u, v, scale=1, scaleratio=0.5)

#         exp_fig_head = [
#             (
#                 0.5,
#                 0.5883883476483185,
#                 None,
#                 1.0,
#                 1.1118033988749896,
#                 None,
#                 1.5,
#                 1.6185854122563141,
#                 None,
#                 2.0,
#             ),
#             (
#                 0.5,
#                 0.6767766952966369,
#                 None,
#                 0.5,
#                 0.6118033988749895,
#                 None,
#                 0.5,
#                 0.5790569415042095,
#                 None,
#                 0.5,
#             ),
#         ]

#         fig_head = [fig["data"][0]["x"][:10], fig["data"][0]["y"][:10]]

#         self.assertEqual(fig_head, exp_fig_head)


# class TestTernarycontour(NumpyTestUtilsMixin, TestCaseNoTemplate):
# def test_wrong_coordinates(self):
#     a, b = np.mgrid[0:1:20j, 0:1:20j]
#     a = a.ravel()
#     b = b.ravel()
#     z = a * b
#     with self.assertRaises(
#         ValueError, msg="Barycentric coordinates should be positive."
#     ):
#         _ = ff.create_ternary_contour(np.stack((a, b)), z)
#     mask = a + b <= 1.0
#     a = a[mask]
#     b = b[mask]
#     with self.assertRaises(ValueError):
#         _ = ff.create_ternary_contour(np.stack((a, b, a, b)), z)
#     with self.assertRaises(ValueError, msg="different number of values and points"):
#         _ = ff.create_ternary_contour(
#             np.stack((a, b, 1 - a - b)), np.concatenate((z, [1]))
#         )
#     # Different sums for different points
#     c = a
#     with self.assertRaises(ValueError):
#         _ = ff.create_ternary_contour(np.stack((a, b, c)), z)
#     # Sum of coordinates is different from one but is equal
#     # for all points.
#     with self.assertRaises(ValueError):
#         _ = ff.create_ternary_contour(np.stack((a, b, 2 - a - b)), z)

# def test_simple_ternary_contour(self):
#     a, b = np.mgrid[0:1:20j, 0:1:20j]
#     mask = a + b < 1.0
#     a = a[mask].ravel()
#     b = b[mask].ravel()
#     c = 1 - a - b
#     z = a * b * c
#     fig = ff.create_ternary_contour(np.stack((a, b, c)), z)
#     fig2 = ff.create_ternary_contour(np.stack((a, b)), z)
#     np.testing.assert_array_almost_equal(
#         fig2["data"][0]["a"], fig["data"][0]["a"], decimal=3
#     )

# def test_colorscale(self):
#     a, b = np.mgrid[0:1:20j, 0:1:20j]
#     mask = a + b < 1.0
#     a = a[mask].ravel()
#     b = b[mask].ravel()
#     c = 1 - a - b
#     z = a * b * c
#     z /= z.max()
#     fig = ff.create_ternary_contour(np.stack((a, b, c)), z, showscale=True)
#     fig2 = ff.create_ternary_contour(
#         np.stack((a, b, c)), z, showscale=True, showmarkers=True
#     )
#     assert isinstance(fig.data[-1]["marker"]["colorscale"], tuple)
#     assert isinstance(fig2.data[-1]["marker"]["colorscale"], tuple)
#     assert fig.data[-1]["marker"]["cmax"] == 1
#     assert fig2.data[-1]["marker"]["cmax"] == 1

# def check_pole_labels(self):
#     a, b = np.mgrid[0:1:20j, 0:1:20j]
#     mask = a + b < 1.0
#     a = a[mask].ravel()
#     b = b[mask].ravel()
#     c = 1 - a - b
#     z = a * b * c
#     pole_labels = ["A", "B", "C"]
#     fig = ff.create_ternary_contour(np.stack((a, b, c)), z, pole_labels=pole_labels)
#     assert fig.layout.ternary.aaxis.title.text == pole_labels[0]
#     assert fig.data[-1].hovertemplate[0] == pole_labels[0]

# def test_optional_arguments(self):
#     a, b = np.mgrid[0:1:20j, 0:1:20j]
#     mask = a + b <= 1.0
#     a = a[mask].ravel()
#     b = b[mask].ravel()
#     c = 1 - a - b
#     z = a * b * c
#     ncontours = 7
#     args = [
#         dict(showmarkers=False, showscale=False),
#         dict(showmarkers=True, showscale=False),
#         dict(showmarkers=False, showscale=True),
#         dict(showmarkers=True, showscale=True),
#     ]

#     for arg_set in args:
#         fig = ff.create_ternary_contour(
#             np.stack((a, b, c)),
#             z,
#             interp_mode="cartesian",
#             ncontours=ncontours,
#             **arg_set,
#         )
#         # This test does not work for ilr interpolation
#         print(len(fig.data))
#         assert len(fig.data) == ncontours + 2 + arg_set["showscale"]


class TestHexbinMapbox(NumpyTestUtilsMixin, TestCaseNoTemplate):
    # def test_aggregation(self):

    #     lat = [0, 1, 1, 2, 4, 5, 1, 2, 4, 5, 2, 3, 2, 1, 5, 3, 5]
    #     lon = [1, 2, 3, 3, 0, 4, 5, 0, 5, 3, 1, 5, 4, 0, 1, 2, 5]
    #     color = np.ones(len(lat))

    #     fig1 = ff.create_hexbin_mapbox(lat=lat, lon=lon, nx_hexagon=1)

    #     actual_geojson = {
    #         "type": "FeatureCollection",
    #         "features": [
    #             {
    #                 "type": "Feature",
    #                 "id": "-8.726646259971648e-11,-0.031886255679892235",
    #                 "geometry": {
    #                     "type": "Polygon",
    #                     "coordinates": [
    #                         [
    #                             [-5e-09, -4.7083909316316985],
    #                             [2.4999999999999996, -3.268549270944215],
    #                             [2.4999999999999996, -0.38356933397072673],
    #                             [-5e-09, 1.0597430482129082],
    #                             [-2.50000001, -0.38356933397072673],
    #                             [-2.50000001, -3.268549270944215],
    #                             [-5e-09, -4.7083909316316985],
    #                         ]
    #                     ],
    #                 },
    #             },
    #             {
    #                 "type": "Feature",
    #                 "id": "-8.726646259971648e-11,0.1192636916419258",
    #                 "geometry": {
    #                     "type": "Polygon",
    #                     "coordinates": [
    #                         [
    #                             [-5e-09, 3.9434377827164666],
    #                             [2.4999999999999996, 5.381998306154031],
    #                             [2.4999999999999996, 8.248045720432454],
    #                             [-5e-09, 9.673766164509932],
    #                             [-2.50000001, 8.248045720432454],
    #                             [-2.50000001, 5.381998306154031],
    #                             [-5e-09, 3.9434377827164666],
    #                         ]
    #                     ],
    #                 },
    #             },
    #             {
    #                 "type": "Feature",
    #                 "id": "0.08726646268698293,-0.031886255679892235",
    #                 "geometry": {
    #                     "type": "Polygon",
    #                     "coordinates": [
    #                         [
    #                             [5.0000000049999995, -4.7083909316316985],
    #                             [7.500000009999999, -3.268549270944215],
    #                             [7.500000009999999, -0.38356933397072673],
    #                             [5.0000000049999995, 1.0597430482129082],
    #                             [2.5, -0.38356933397072673],
    #                             [2.5, -3.268549270944215],
    #                             [5.0000000049999995, -4.7083909316316985],
    #                         ]
    #                     ],
    #                 },
    #             },
    #             {
    #                 "type": "Feature",
    #                 "id": "0.08726646268698293,0.1192636916419258",
    #                 "geometry": {
    #                     "type": "Polygon",
    #                     "coordinates": [
    #                         [
    #                             [5.0000000049999995, 3.9434377827164666],
    #                             [7.500000009999999, 5.381998306154031],
    #                             [7.500000009999999, 8.248045720432454],
    #                             [5.0000000049999995, 9.673766164509932],
    #                             [2.5, 8.248045720432454],
    #                             [2.5, 5.381998306154031],
    #                             [5.0000000049999995, 3.9434377827164666],
    #                         ]
    #                     ],
    #                 },
    #             },
    #             {
    #                 "type": "Feature",
    #                 "id": "0.04363323129985823,0.04368871798101678",
    #                 "geometry": {
    #                     "type": "Polygon",
    #                     "coordinates": [
    #                         [
    #                             [2.4999999999999996, -0.38356933397072673],
    #                             [5.0000000049999995, 1.0597430482129082],
    #                             [5.0000000049999995, 3.9434377827164666],
    #                             [2.4999999999999996, 5.381998306154031],
    #                             [-5.0000001310894304e-09, 3.9434377827164666],
    #                             [-5.0000001310894304e-09, 1.0597430482129082],
    #                             [2.4999999999999996, -0.38356933397072673],
    #                         ]
    #                     ],
    #                 },
    #             },
    #         ],
    #     }

    #     actual_agg = [2.0, 2.0, 1.0, 3.0, 9.0]

    #     self.assert_dict_equal(fig1.data[0].geojson, actual_geojson)
    #     assert np.array_equal(fig1.data[0].z, actual_agg)

    #     fig2 = ff.create_hexbin_mapbox(
    #         lat=lat,
    #         lon=lon,
    #         nx_hexagon=1,
    #         color=color,
    #         agg_func=np.mean,
    #     )

    #     assert np.array_equal(fig2.data[0].z, np.ones(5))

    #     fig3 = ff.create_hexbin_mapbox(
    #         lat=np.random.randn(1000),
    #         lon=np.random.randn(1000),
    #         nx_hexagon=20,
    #     )

    #     assert fig3.data[0].z.sum() == 1000

    def test_build_dataframe(self):
        np.random.seed(0)
        N = 10000
        nx_hexagon = 20
        n_frames = 3

        lat = np.random.randn(N)
        lon = np.random.randn(N)
        color = np.ones(N)
        frame = np.random.randint(0, n_frames, N)
        df = pd.DataFrame(
            np.c_[lat, lon, color, frame],
            columns=["Latitude", "Longitude", "Metric", "Frame"],
        )

        fig1 = ff.create_hexbin_mapbox(lat=lat, lon=lon, nx_hexagon=nx_hexagon)
        fig2 = ff.create_hexbin_mapbox(
            data_frame=df, lat="Latitude", lon="Longitude", nx_hexagon=nx_hexagon
        )

        assert isinstance(fig1, go.Figure)
        assert len(fig1.data) == 1
        self.assert_dict_equal(
            fig1.to_plotly_json()["data"][0], fig2.to_plotly_json()["data"][0]
        )

        fig3 = ff.create_hexbin_mapbox(
            lat=lat,
            lon=lon,
            nx_hexagon=nx_hexagon,
            color=color,
            agg_func=np.sum,
            min_count=0,
        )
        fig4 = ff.create_hexbin_mapbox(
            lat=lat,
            lon=lon,
            nx_hexagon=nx_hexagon,
            color=color,
            agg_func=np.sum,
        )
        fig5 = ff.create_hexbin_mapbox(
            data_frame=df,
            lat="Latitude",
            lon="Longitude",
            nx_hexagon=nx_hexagon,
            color="Metric",
            agg_func=np.sum,
        )

        self.assert_dict_equal(
            fig1.to_plotly_json()["data"][0], fig3.to_plotly_json()["data"][0]
        )
        self.assert_dict_equal(
            fig4.to_plotly_json()["data"][0], fig5.to_plotly_json()["data"][0]
        )

        fig6 = ff.create_hexbin_mapbox(
            data_frame=df,
            lat="Latitude",
            lon="Longitude",
            nx_hexagon=nx_hexagon,
            color="Metric",
            agg_func=np.sum,
            animation_frame="Frame",
        )

        fig7 = ff.create_hexbin_mapbox(
            lat=lat,
            lon=lon,
            nx_hexagon=nx_hexagon,
            color=color,
            agg_func=np.sum,
            animation_frame=frame,
        )

        assert len(fig6.frames) == n_frames
        assert len(fig7.frames) == n_frames
        assert fig6.data[0].geojson == fig1.data[0].geojson
