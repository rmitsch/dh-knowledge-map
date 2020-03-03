import pandas as pd
from typing import Tuple, List, Dict
import numpy as np
import umap
from scipy.spatial.distance import cdist


def load_data(storage_path: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Loads datasets.
    :param storage_path:
    :type storage_path:
    :return: As dataframes: Courses, countries, disciplines, universities, Tadirah techniques, Tadirah objects.
    """

    courses: pd.DataFrame = pd.read_json(storage_path + "courses.json")
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

    return (
        courses,
        pd.read_json(storage_path + "countries.json"),
        pd.read_json(storage_path + "disciplines.json"),
        pd.read_json(storage_path + "universities.json"),
        tadirah_techniques,
        tadirah_objects
    )


def compute_initial_positions(knowledge_entities: pd.DataFrame) -> pd.DataFrame:
    """
    Compute initial positions for all courses and knowledge entities.
    :param knowledge_entities:
    :return: Dataframe with coordinates per course and knowledge entity.
    """

    # Get matrix with knowledge entity flags as features.
    course_features: pd.DataFrame = knowledge_entities.drop(columns=["id", "description"]).pivot_table(
        index="course_id", columns="guid", aggfunc=lambda x: True
    ).fillna(False).astype(int)
    knowledge_entity_ids: list = [idx[1] for idx in course_features.columns.values]

    # todo create similarity/edge strength matrix between KEs by computing overlap of features as defined by courses
    # todo compute distance matrix so that only overlaps of 1 count as similarity. i.e.: all pairs start of with maximal
    #  distance and lose 1 for every 1-match. -> similar (categorical) distance function offered in sklearn?
    #  -->
    #  1. implement and test adapted distance function. if reasonable placements: keep embedding coordinates as basis
    #  for node placement, both for KEs as for Cs.
    #  2. generate co-occurence matrix between KEs (transpose matrix from 1.). compute distance matrix for KEs with
    #  abovementioned distance metric for KEs.
    #  3. compute edge strength based on distance matrix from 2.
    #  4. evaluate results.
    #  5. refine design (sidebar? node placement acceptable?); plan interactive steps.
    #  6. implement interactive steps.
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    #     print(course_features.head())
    #     print(course_features.T.head())
    #     print(knowledge_entities.head())
    #     print(knowledge_entities.drop(columns=["id", "description"]).pivot_table(
    #         index="guid", columns="course_id", aggfunc=lambda x: 1
    #     ))
    #
    # exit()
    # Reduce dimensionality.
    dimred: umap.UMAP = umap.UMAP(n_components=2, n_neighbors=4, metric="jaccard")
    embedding: pd.DataFrame = pd.DataFrame(
        dimred.fit_transform(
            np.concatenate([course_features.values, np.identity(len(knowledge_entity_ids))])
        ),
        columns=["x", "y"]
    )
    embedding["id"] = np.concatenate([course_features.index.values, knowledge_entity_ids])

    # Normalize coordinates to between 0 and 1.
    embedding.x = (embedding.x - embedding.x.min()) / (embedding.x.max() - embedding.x.min())
    embedding.y = (embedding.y - embedding.y.min()) / (embedding.y.max() - embedding.y.min())

    return embedding.set_index("id")


def create_network(
        courses: pd.DataFrame,
        tadirah_objects: pd.DataFrame,
        tadirah_technologies: pd.DataFrame,
        embedding: pd.DataFrame,
        plot_size: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates network suitable for cytoscape.js from provided dataframes.
    :param courses:
    :param tadirah_objects:
    :param tadirah_technologies:
    :param embedding:
    :param plot_size:
    :return: Tuple with (1) list of nodes and (2) list of edges.
    """

    # Discard Tadirah entries related to courses not showing in courses DF.
    tadirah_technologies = tadirah_technologies[tadirah_technologies.course_id.isin(courses.index)]
    tadirah_objects = tadirah_objects[tadirah_objects.course_id.isin(courses.index)]
    # Discard courses without Tadira entries.
    courses = courses[
        (courses.index.isin(tadirah_objects.course_id)) |
        (courses.index.isin(tadirah_technologies.course_id))
    ]

    def convert_tadirah_data_to_graph_dicts(df: pd.DataFrame, tadirah_type: str) -> np.ndarray:
        """
        Converts Tadirah information into array of dicts.
        :param df:
        :param tadirah_type:
        :return: Tadirah information as array of dicts.
        """
        assert tadirah_type in ("object", "technology")
        id_col: str = "tadirah_" + tadirah_type + "_id"
        type_shortened: str = tadirah_type.upper()[0]

        return courses.join(
            df.set_index("course_id"), lsuffix="_course", rsuffix="_object", how="inner"
        )[["id"]].reset_index().rename(columns={"index": "course_id", "id": id_col}).apply(
            lambda row: {
                "data": {
                    "source": "T" + type_shortened + str(int(row[id_col])),
                    "target": "C" + str(int(row.course_id)),
                    "weight": 1
                }
            },
            axis=1
        ).values

    # import matplotlib.pyplot as plt
    # embedding.plot.scatter(x="x", y="y")
    # plt.show()

    return (
        # Nodes.
        # todo add initial positions from embeddings - adjust values for screen size!
        np.concatenate([
            # Course nodes.
            courses.apply(
                lambda row: {
                    "data": {"id": "C" + str(row.name), "label": row["name"]},
                    "position": {
                        "x": embedding.loc[str(row.name)].x * plot_size["width"],
                        "y": embedding.loc[str(row.name)].y * plot_size["height"]
                    }
                }, axis=1
            ).values,
            # Tadirah object nodes.
            tadirah_objects.apply(
                lambda row: {
                    "data": {"id": "TO" + str(row["id"]), "label": row["name"]},
                    "position": {
                        "x": embedding.loc[str(row.guid)].x * plot_size["width"],
                        "y": embedding.loc[str(row.guid)].y * plot_size["height"]
                    }
                }, axis=1
            ).values,
            # Tadirah technology nodes.
            tadirah_technologies.apply(
                lambda row: {
                    "data": {"id": "TT" + str(row["id"]), "label": row["name"]},
                    "position": {
                        "x": embedding.loc[str(row.guid)].x * plot_size["width"],
                        "y": embedding.loc[str(row.guid)].y * plot_size["height"]
                    }
                }, axis=1
            ).values
        ]),
        # Edges.
        np.concatenate([
            # Edges from course to Tadirah object nodes.
            convert_tadirah_data_to_graph_dicts(tadirah_objects, "object"),
            # Edges from course to Tadirah technology nodes.
            convert_tadirah_data_to_graph_dicts(tadirah_technologies, "technology")
        ])
    )
