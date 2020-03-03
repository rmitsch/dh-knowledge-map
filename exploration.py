import pandas as pd
import utils
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import numpy as np

# todo
#   - compute initial course positions / experiment w/ layouts
#   - usage mode? do we start with courses and expand their knowledge entities? the other way around? do we present
#     everything?


storage_path: str = "/home/raphael/Development/data/DHH/"

courses, countries, disciplines, universities, tadirah_techniques, tadirah_objects = utils.load_data(storage_path)
courses = courses.drop(columns=[
    "tadirah_techniques", "tadirah_objects", "deletion_reason_id", "deletion_reason", "course_parent_type",
    "course_duration_unit", "course_duration_unit_id", "duration", "lon", "lat"
]).set_index("id")
# courses = courses.head(100)

plot_size: dict = {"width": 1300, "height": 650}
network: tuple = utils.create_network(
    courses,
    tadirah_objects,
    tadirah_techniques,
    utils.compute_initial_positions(pd.concat([tadirah_techniques, tadirah_objects])),
    plot_size
)

cyto.load_extra_layouts()
app = dash.Dash(__name__)
app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape',
        elements=network[0].tolist() + network[1].tolist(),
        layout={'name': 'preset'},  # cose
        style={'width': str(plot_size["width"]) + 'px', 'height': str(plot_size["height"]) + 'px'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    "width": 3,
                    "height": 3,
                    "font-size": 3,
                    "background-color": "green"
                }
            },
            {
                'selector': 'edge',
                'style': {
                    "width": 0.3,
                    "opacity": 0.3
                }
            },
            {
                'selector': '[id ^= "TO"]',
                'style': {
                    'background-color': '#FF4136',
                    'shape': 'rectangle'
                }
            },
            {
                'selector': '[id ^= "TT"]',
                'style': {
                    'background-color': 'blue',
                    'shape': 'rectangle'
                }
            }
        ]
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
