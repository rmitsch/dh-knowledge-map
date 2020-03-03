import pandas as pd
import utils
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import json

# todo
#   - usage mode? do we start with courses and expand their knowledge entities? the other way around? do we present
#     everything?


storage_path: str = "/home/raphael/Development/data/DHH/"
(
    courses, countries, disciplines, universities, tadirah_techniques, tadirah_objects, tadirah_techniques_counts,
    tadirah_objects_counts
) = utils.load_data(storage_path)

last_guid: str = ""
plot_size: dict = {"width": 1300, "height": 400}
embedding: pd.DataFrame = utils.compute_embedding(pd.concat([tadirah_techniques, tadirah_objects]))
tadirah_objects = tadirah_objects.set_index("guid")
tadirah_techniques = tadirah_techniques.set_index("guid")

# guid = "TT12"
# local_network = utils.create_network(
#     courses[(courses.index.isin(tadirah_techniques.loc[[guid]].course_id))],
#     tadirah_objects.head(0),
#     tadirah_techniques.loc[[guid]],
#     tadirah_objects_counts,
#     tadirah_techniques_counts,
#     embedding,
#     plot_size
# )
# exit()
# local_network = utils.create_network(
#     courses.loc[[int(408)]],
#     tadirah_objects,
#     tadirah_techniques,
#     embedding,
#     plot_size
# )
# exit()

cyto.load_extra_layouts()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app: dash.Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    [
        html.Div(
            utils.create_global_scatterplot(embedding, courses, tadirah_objects, tadirah_techniques),
            style={"display": "inline-block", "width": "49%", "height": "90%"}
        ),
        html.Div(
            utils.create_cytoscape_graph(plot_size),
            style={"display": "inline-block", "width": "49%", "height": "90%"}
        ),
        html.Div(
            [
                html.Pre(id='click-data', style={
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                })
            ],
            style={"width": "100%", "height": "10%"},
            className='three columns'
        )
    ],
    style={"display": "inline-block", "width": "100%", "height": "95%"}
)


@app.callback(
    Output('click-data', 'children'),
    [Input('basic-interactions', 'clickData')]
)
def display_click_data(clickdata_input):
    return json.dumps(clickdata_input, indent=2)


@app.callback(
    Output('cytoscape-elements-callbacks', 'elements'),
    [Input('basic-interactions', 'clickData')],
    [State('cytoscape-elements-callbacks', 'elements')]
)
def display_click_data(clickdata_input, elements):
    if clickdata_input is not None:
        global last_guid
        guid: str = str(clickdata_input["points"][0]["customdata"])
        if last_guid != guid:
            last_guid = guid
            print("guid:", guid)
            local_network: tuple = None

            if guid.startswith("TT"):
                local_network = utils.create_network(
                    courses[(courses.index.isin(tadirah_techniques.loc[[guid]].course_id))],
                    tadirah_objects.head(0),
                    tadirah_techniques.loc[[guid]],
                    tadirah_objects_counts,
                    tadirah_techniques_counts,
                    embedding,
                    plot_size
                )

            elif guid.startswith("TO"):
                local_network = utils.create_network(
                    courses[(courses.index.isin(tadirah_objects.loc[[guid]].course_id))],
                    tadirah_objects.loc[[guid]],
                    tadirah_techniques.head(0),
                    tadirah_objects_counts,
                    tadirah_techniques_counts,
                    embedding,
                    plot_size
                )
            else:
                local_network = utils.create_network(
                    courses.loc[[int(guid)]],
                    tadirah_objects,
                    tadirah_techniques,
                    tadirah_objects_counts,
                    tadirah_techniques_counts,
                    embedding,
                    plot_size
                )

            return local_network[0].tolist() + local_network[1].tolist()

    else:
        return []


if __name__ == '__main__':
    app.run_server(debug=True)
