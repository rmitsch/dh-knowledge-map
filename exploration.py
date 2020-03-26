import pandas as pd
import utils
import dash
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output, State
import uuid
import json
import dash_core_components as dcc
from wikification import integrate_wiki, find_related_wiki
from utils import fetch_dh_registry_data

# Keep track of clicked elements.
extended_nodes: set = set()

# (Down-)load DH registriy data.
# absolute_path: str = os.path.dirname(os.path.abspath(__file__))
storage_directory: str = "/tmp"
fetch_dh_registry_data(storage_directory)
(
    courses, countries, disciplines, universities, tadirah_techniques, tadirah_objects, tadirah_techniques_counts,
    tadirah_objects_counts
) = utils.load_data(storage_directory)
filtered_courses: pd.DataFrame = courses

last_guid: str = ""
embedding: pd.DataFrame = utils.compute_embedding(pd.concat([tadirah_techniques, tadirah_objects]))
tadirah_objects = tadirah_objects.set_index("guid")
tadirah_techniques = tadirah_techniques.set_index("guid")

cyto.load_extra_layouts()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app: dash.Dash = dash.Dash(__name__)  # , external_stylesheets=external_stylesheets
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id='scatterplot',
                        style={"height": "100%"},
                        figure=utils.create_global_scatterplot_figure(
                            utils.preprocess_data_for_global_scatterplot(
                                embedding, courses, tadirah_objects, tadirah_techniques
                            )
                        ),
                    ),
                    style={"display": "inline-block", "width": "49%", "height": "100%"}
                ),
                html.Div(
                    utils.create_network_graph(),
                    style={"display": "inline-block", "width": "49%", "height": "100%"}
                ),
            ],
            style={"display": "block", "width": "100%", "height": "65%", "overflow": "hidden"}
        ),
        html.Div(
            [
                html.Div(
                    utils.create_courses_table(courses),
                    style={
                        "width": "49.5%",
                        "height": "100%",
                        "display": "inline-block"
                    }
                ),
                html.Div(
                    [
                        html.Pre(id='graph-click-data', style={
                            'border': 'thin lightgrey solid',
                            'overflowX': 'scroll',
                            'overflowY': 'scroll'
                        })
                    ],
                    style={
                        "width": "49.5%",
                        "height": "100%",
                        "display": "inline-block",
                        "float": "right"
                    }
                )
            ],
            style={"width": "100%", "height": "35%", "display": "block", "overflowY": "hidden", "overflowX": "hidden"}
        )
    ],
    style={"display": "inline-block", "width": "100%", "height": "calc(100vh - 20px)"}
)


# @app.callback(
#     Output('courses-table', 'children'),
#     [Input('scatterplot', 'clickData')]
# )
# def display_click_data(clickdata_input):
#     if clickdata_input:
#         return json.dumps(clickdata_input, indent=2)


@app.callback(
    Output('scatterplot', "figure"),
    [
        Input('courses-table', "derived_virtual_data"),
        Input('courses-table', "derived_virtual_selected_rows")
    ]
)
def update_scatterplot(rows: list, derived_virtual_selected_rows: list):
    """
    Update scatterplot after rows in course table have been selected.
    :param rows:
    :param derived_virtual_selected_rows:
    :return:
    """

    global filtered_courses

    if rows is not None and derived_virtual_selected_rows is not None:
        if len(derived_virtual_selected_rows) > 0:
            selected_course_ids: set = {rows[ix]["id"] for ix in derived_virtual_selected_rows}
            filtered_courses = courses.loc[selected_course_ids]
        else:
            filtered_courses = courses

        return utils.create_global_scatterplot_figure(
            utils.preprocess_data_for_global_scatterplot(
                embedding, filtered_courses, tadirah_objects, tadirah_techniques
            )
        )


@app.callback(
    Output('network-graph', 'elements'),
    [
        Input('scatterplot', 'clickData'),
        Input('network-graph', 'tapNodeData')
    ],
    [State('network-graph', 'elements')]
)
def update_network_graph(scatterplot_input: dict, networkgraph_input: dict, elements: list):
    """
    Updates network graph after relevant input in either scatterplot or network graph.
    :param scatterplot_input:
    :param networkgraph_input:
    :param elements:
    """

    fc: pd.DataFrame = filtered_courses

    if scatterplot_input is not None:
        global last_guid
        guid: str = str(scatterplot_input["points"][0]["customdata"])

        if last_guid != guid:
            last_guid = guid
            local_network: tuple = None

            # Show courses for chosen Tadira technique.
            if guid.startswith("TT"):
                local_network = utils.create_network(
                    fc[(fc.index.isin(tadirah_techniques.loc[[guid]].course_id))],
                    tadirah_objects.head(0),
                    tadirah_techniques.loc[[guid]],
                    tadirah_objects_counts,
                    tadirah_techniques_counts,
                    embedding
                )

            # Show courses for chosen Tadira object.
            elif guid.startswith("TO"):
                local_network = utils.create_network(
                    fc[(fc.index.isin(tadirah_objects.loc[[guid]].course_id))],
                    tadirah_objects.loc[[guid]],
                    tadirah_techniques.head(0),
                    tadirah_objects_counts,
                    tadirah_techniques_counts,
                    embedding
                )

            else:
                local_network = utils.create_network(
                    fc.loc[[int(guid)]],
                    tadirah_objects,
                    tadirah_techniques,
                    tadirah_objects_counts,
                    tadirah_techniques_counts,
                    embedding
                )

            return local_network[0].tolist() + local_network[1].tolist()

    if networkgraph_input is not None:
        global extended_nodes

        if networkgraph_input["id"] not in extended_nodes:
            related_wiki = find_related_wiki(networkgraph_input)
            # todo collapse node if clicked the second time

            for wiki in related_wiki:
                uuid_wiki: str = str(uuid.uuid1())
                elements.extend([{
                    'data': {'id': 'W{}'.format(uuid_wiki), 'label': wiki},
                },
                    {'data': {'source': 'W{}'.format(uuid_wiki), 'target': networkgraph_input["id"], 'weight': 0.5,
                              'id': str(uuid.uuid1())}}
                ])

            # Remember this node has been extended.
            extended_nodes.add(networkgraph_input["id"])

        return elements

    # todo wikipedia.exceptions.DisambiguationError: "lola" may refer to:
    else:
        return []


# @app.callback(
#     Output('graph-click-data', 'children'),
#     [Input('network-graph', 'tapNodeData')]
# )
# def on_click_in_graph(data):
#     if data:
#         return integrate_wiki(data)


if __name__ == '__main__':
    app.run_server(debug=True)
