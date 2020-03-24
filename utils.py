import json
import os

import pandas as pd
from typing import Tuple, List, Dict
import numpy as np
import requests
import umap
from scipy.spatial.distance import cdist
from scipy.special import expit
import os
import sys
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import itertools

from constants import DH_REGISTRY_MATERIALS_ENDPOINTS

absolute_path = os.path.dirname(os.path.abspath(__file__))

def fetch_dh_registry_data():
    """
    Download data from DH Registry API
    :return: As json objects: Courses, countries, disciplines, universities.
    """
    for endpoint in DH_REGISTRY_MATERIALS_ENDPOINTS.keys():
        response = requests.get(DH_REGISTRY_MATERIALS_ENDPOINTS[endpoint]).json()
        with open(os.path.join(absolute_path, "{}.json".format(endpoint)), 'w') as data_dump:
            json.dump(response, data_dump)


def upload_json_data(path, file_name):
    open_file = open(os.path.join(path, "{}.json".format(file_name)))
    json_data = json.load(open_file)
    return json_data

def load_data(storage_path: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Loads datasets.
    :param storage_path:
    :type storage_path:
    :return: As dataframes: Courses, countries, disciplines, universities, Tadirah techniques, Tadirah objects, counts
    of Tadirah techniques per course, counts of Tadirah objects per course.
    """

    json_data_courses = upload_json_data(absolute_path, "courses")
    courses = pd.DataFrame(json_data_courses)
    courses = courses[courses.active == True]
    tadirah_techniques: pd.DataFrame = pd.DataFrame([
        {**{"course_id": row[1].id}, **technique}
        for row in courses[["id", "tadirah_techniques"]].iterrows()
        for technique in row[1].tadirah_techniques
    ])
    tadirah_objects: pd.DataFrame = pd.DataFrame([
        {**{"course_id": row[1].id}, **obj}
        for row in courses[["id", "tadirah_objects"]].iterrows()
        for obj in row[1].tadirah_objects
    ])
    tadirah_techniques["guid"] = "TT" + tadirah_techniques["id"].astype(str)
    tadirah_objects["guid"] = "TO" + tadirah_objects["id"].astype(str)

    courses = courses.drop(columns=[
        "tadirah_techniques", "tadirah_objects", "deletion_reason_id", "deletion_reason", "course_parent_type",
        "course_duration_unit", "course_duration_unit_id", "duration", "lon", "lat"
    ]).set_index("id")

    tadirah_techniques_counts: pd.DataFrame = tadirah_techniques.merge(
        courses[["name"]], left_on="course_id", right_on="id"
    ).groupby("course_id").count()[["id"]].rename(columns={"id": "cnt"})
    tadirah_objects_counts: pd.DataFrame = tadirah_objects.merge(
        courses[["name"]], left_on="course_id", right_on="id"
    ).groupby("course_id").count()[["id"]].rename(columns={"id": "cnt"})

    return (
        courses,
        pd.read_json(json.dumps(upload_json_data(absolute_path, "courses"))),
        pd.read_json(json.dumps(upload_json_data(absolute_path, "disciplines"))),
        pd.read_json(json.dumps(upload_json_data(absolute_path, "universities"))),
        tadirah_techniques,
        tadirah_objects,
        tadirah_techniques_counts,
        tadirah_objects_counts
    )


def compute_embedding(knowledge_entities: pd.DataFrame, invalidate_cache: bool = False) -> pd.DataFrame:
    """
    Compute initial positions for all courses and knowledge entities.
    :param knowledge_entities:
    :param invalidate_cache:
    :return: Dataframe with coordinates per course and knowledge entity.
    """

    cache_path: str = "/tmp/dhh-embedding.pkl"

    if not os.path.isfile(cache_path) or invalidate_cache:
        # Get matrix with knowledge entity flags as features.
        course_features: pd.DataFrame = knowledge_entities.drop(columns=["id", "description"]).pivot_table(
            index="course_id", columns="guid", aggfunc=lambda x: True
        ).fillna(False).astype(int)
        knowledge_entity_ids: list = [idx[1] for idx in course_features.columns.values]

        # Reduce dimensionality.
        dimred: umap.UMAP = umap.UMAP(n_components=2, n_neighbors=4, metric="jaccard")
        embedding: pd.DataFrame = pd.DataFrame(
            dimred.fit_transform(
                np.concatenate([course_features.values, np.identity(len(knowledge_entity_ids))])
            ),
            columns=["x", "y"]
        )
        embedding["id"] = np.concatenate([course_features.index.values, knowledge_entity_ids])
        embedding["type"] = np.concatenate([["C"] * len(course_features), [keid[:2] for keid in knowledge_entity_ids]])

        # Normalize coordinates to between 0 and 1.
        embedding.x = (embedding.x - embedding.x.min()) / (embedding.x.max() - embedding.x.min())
        embedding.y = (embedding.y - embedding.y.min()) / (embedding.y.max() - embedding.y.min())

        # Cache.
        embedding.to_pickle("/tmp/dhh-embedding.pkl")

    else:
        embedding: pd.DataFrame = pd.read_pickle(cache_path)

    return embedding.set_index("id")

def create_network(
        courses: pd.DataFrame,
        tadirah_objects: pd.DataFrame,
        tadirah_technologies: pd.DataFrame,
        tadirah_objects_counts: pd.DataFrame,
        tadirah_technologies_counts: pd.DataFrame,
        embedding: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates network suitable for cytoscape.js from provided dataframes.
    :param courses:
    :param tadirah_objects:
    :param tadirah_technologies:
    :param tadirah_objects_counts:
    :param tadirah_technologies_counts:
    :param embedding:
    :return: Tuple with (1) list of nodes and (2) list of edges.
    """

    # Discard Tadirah entries related to courses not showing in courses DF.
    tadirah_technologies = tadirah_technologies[tadirah_technologies.course_id.isin(courses.index)]
    tadirah_objects = tadirah_objects[tadirah_objects.course_id.isin(courses.index)]
    knowledge_entities: pd.DataFrame = pd.concat([tadirah_objects, tadirah_technologies])

    # Discard courses without Tadira entries.
    courses = courses[(courses.index.isin(knowledge_entities.course_id))]

    ###################################################
    # Compute weights of edges between KEs and courses.
    ###################################################

    knowledge_entities = knowledge_entities.reset_index().merge(
        tadirah_objects_counts, on="course_id"
    ).merge(
        tadirah_technologies_counts, on="course_id", suffixes=("_tadirah_objs", "_tadirah_techs")
    ).set_index("guid")

    knowledge_entities.course_id = knowledge_entities.course_id.astype(str)
    knowledge_entities["weight"] = np.power(
        1 / (knowledge_entities.cnt_tadirah_objs + knowledge_entities.cnt_tadirah_techs), 0.5
    )
    # Adjust IDs.
    knowledge_entities["source"] = knowledge_entities.index.values
    knowledge_entities["target"] = "C" + knowledge_entities.course_id

    ###################################################
    # Compute weights of edges between courses.
    ###################################################

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        courses_w_edges: pd.DataFrame = pd.DataFrame(
            list(itertools.product(courses.index.values, courses.index.values)),
            columns=["source", "target"]
        )
        courses_w_edges = courses_w_edges[courses_w_edges.source != courses_w_edges.target]
        courses_w_edges.source = courses_w_edges.source.astype(str)
        courses_w_edges.target = courses_w_edges.target.astype(str)
        courses_w_edges = courses_w_edges.reset_index().merge(
            embedding[["x", "y"]], left_on="source", right_on="id", how="inner"
        ).merge(
            embedding[["x", "y"]], left_on="target", right_on="id", how="inner", suffixes=("_source", "_target")
        )
        courses_w_edges["weight"] = 1 / np.sqrt(
            np.power(courses_w_edges.x_source - courses_w_edges.x_target, 2) +
            np.power(courses_w_edges.y_source - courses_w_edges.y_target, 2)
        )

        # Normalize for translation into opacity values.
        courses_w_edges.weight -= courses_w_edges.weight.min()
        courses_w_edges.weight /= courses_w_edges.weight.max()
        # Adjust IDs.
        courses_w_edges["source"] = "C" + courses_w_edges.source
        courses_w_edges["target"] = "C" + courses_w_edges.target

    ###################################################
    # Assemble dataset with nodes and edges.
    ###################################################

    def extract_knowledge_entities_graph_edges(df: pd.DataFrame, min_weight: float = 0) -> np.ndarray:
        """
        Converts knowledge entity similarities into array of dicts.
        :param df: Number of connections between knowledge entities as adjacency matrix.
        :param min_weight: Minimal weight for an edge to not be discarded.
        :return: KE similarities as array of dicts.
        """

        if len(df) == 0:
            return np.asarray([])

        df.weight = np.power(df.weight, 1)
        return df[df.weight > min_weight].apply(
            lambda row: {"data": {"source": row["source"], "target": row["target"], "weight": row["weight"]}}, axis=1
        ).values

    # Remove redundant entries/edges for KE nodes.
    grouped_kes: pd.DataFrame = knowledge_entities.groupby(["guid", "name"])["course_id"].apply(list).reset_index()

    return (
        # Nodes.
        np.concatenate([
            # Course nodes.
            courses.apply(
                lambda row: {
                    "data": {"id": "C" + str(row.name), "label": row["name"]},
                    "position": {
                        # Note: If "preset" is to be used as graph layout, coordinates have to be scaled with plot
                        # dimensions here. Ignored for now since most likely a graph layouting algorithm will be used.
                        "x": embedding.loc[str(row.name)].x,
                        "y": embedding.loc[str(row.name)].y
                    }
                }, axis=1
            ).values,
            # KE nodes.
            grouped_kes.apply(
                lambda row: {
                    "data": {"id": row["guid"], "label": row["name"]},
                    "position": {
                        # Note: If "preset" is to be used as graph layout, coordinates have to be scaled with plot
                        # dimensions here. Ignored for now since most likely a graph layouting algorithm will be used.
                        "x": embedding.loc[str(row.guid)].x,
                        "y": embedding.loc[str(row.guid)].y
                    }
                }, axis=1
            ).values
        ]),
        # Edges.
        np.concatenate([
            # Edges between knowledge entities.
            extract_knowledge_entities_graph_edges(knowledge_entities),
            # Edges between courses.
            extract_knowledge_entities_graph_edges(courses_w_edges, min_weight=0.1)
        ])
    )


def create_global_scatterplot(
        embedding: pd.DataFrame, courses: pd.DataFrame, tadirah_objects: pd.DataFrame, tadirah_techniques: pd.DataFrame
) -> dcc.Graph:
    """
    Defines configuration for global scatterplot.
    :param embedding:
    :param courses:
    :param tadirah_objects:
    :param tadirah_techniques:
    :return: dcc.Graph object for local scatter plot.
    """

    grouped_tos: pd.DataFrame = tadirah_objects.groupby(
        ["guid", "name"]
    )["course_id"].apply(list).reset_index().set_index("guid")
    grouped_tts: pd.DataFrame = tadirah_techniques.groupby(
        ["guid", "name"]
    )["course_id"].apply(list).reset_index().set_index("guid")

    return dcc.Graph(
        id='basic-interactions',
        style={"height": "100%"},
        figure={
            'data': [
                {
                    'x': embedding[embedding.type == "C"].x,
                    'y': embedding[embedding.type == "C"].y,
                    'text': courses.loc[embedding[embedding.type == "C"].index.values.astype(int)].name,
                    'customdata': courses.loc[
                        embedding[embedding.type == "C"].index.values.astype(int)
                    ].index.values,
                    'customdata2': "blub",
                    'name': 'Courses',
                    'mode': 'markers',
                    'marker': {'size': 3}
                },
                {
                    'x': embedding[embedding.type == "TO"].x,
                    'y': embedding[embedding.type == "TO"].y,
                    'text': grouped_tos.loc[
                        embedding[embedding.type == "TO"].index.values
                    ].name,
                    'customdata': grouped_tos.loc[
                        embedding[embedding.type == "TO"].index.values
                    ].index.values,
                    'name': 'Tadirah Objects',
                    'mode': 'markers',
                    'marker': {'size': 5, "symbol": "diamond"}
                },
                {
                    'x': embedding[embedding.type == "TT"].x,
                    'y': embedding[embedding.type == "TT"].y,
                    'text': grouped_tts.loc[
                        embedding[embedding.type == "TT"].index.values
                    ].name,
                    'customdata': grouped_tts.loc[
                        embedding[embedding.type == "TT"].index.values
                    ].index.values,
                    'name': 'Tadirah Techniques',
                    'mode': 'markers',
                    'marker': {'size': 5, "symbol": "diamond"}
                }
            ],
            'layout': {
                'clickmode': 'event+select',
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "margin": {
                    "l": 0,
                    "r": 0,
                    "b": 0,
                    "t": 0,
                    "pad": 0
                }
            }
        }
    )


def create_cytoscape_graph() -> cyto.Cytoscape:
    """
    Creates Cytoscape graph object.
    :return: Cytoscape graph object.
    """
    return cyto.Cytoscape(
        id='cytoscape-elements-callbacks',
        elements=[],  # network[0].tolist() + network[1].tolist(),
        layout={'name': 'cose-bilkent', "animate": True, "fit": True},
        # reasonable: cose-bilkent, cola, euler, circle (?). preset works if coordinates are scaled.
        style={'width': "100%", 'height': "100%"},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    "width": 8,
                    "height": 8,
                    "font-size": 8,
                    "background-color": "blue"
                }
            },
            {
                'selector': 'edge',
                'style': {
                    "curve-style": "bezier",
                    "width": 0.3,
                    "opacity": "data(weight)"
                }
            },
            {
                'selector': '[id ^= "TO"]',
                'style': {
                    'background-color': 'orange',
                    'shape': 'rectangle'
                }
            },
            {
                'selector': '[id ^= "TT"]',
                'style': {
                    'background-color': 'green',
                    'shape': 'rectangle'
                }
            },
            {
                'selector': '[id ^= "C"]',
                'style': {
                    "label": "data(label)",
                    "font-size": 3,
                    "width": 4,
                    "height": 4,
                    "tooltip": "blub"
                }
            }
        ]
    )
