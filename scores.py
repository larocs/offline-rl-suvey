import csv
import os
from collections import defaultdict
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib
import numpy as np

RGBAColor = Tuple[float, float, float, float]
MatplotlibColor = Union[str, RGBAColor]

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

SCORES_CSV = os.path.join("assets", "scores.csv")
CMAP = matplotlib.colormaps["magma"]


class DatasetProperty(Enum):
    NARROW_AND_BIASED_DIST = "NB"
    UNDIRECTED_AND_MULTITASK_DATA = "UM"
    SPARSE_REWARDS = "SR"
    SUBOPTIMAL_DATA = "SD"
    NON_REPRESENTABLE_BEHAVIOR_POLICY = "NR"
    NON_MARKOVIAN_BEHAVIOR_POLICY = "NM"
    REALISTIC_DOMAIN = "RD"
    NON_STATIONARITY = "NS"


class DatasetSuite(Enum):
    MAZE2D = auto()
    ANTMAZE = auto()
    GYM_MUJOCO = auto()
    ADROIT = auto()
    KITCHEN = auto()
    FLOW = auto()
    CARLA = auto()


class TaxonomyClass(str, Enum):
    DIRECT_POLICY_CONSTRAINT = "Direct Policy Constraints"
    IMPLICIT_POLICY_CONSTRAINT = "Implicit Policy Constraints"
    POLICY_REGULARIZATION = "Policy Regularization"
    VALUE_REGULARIZATION = "Value Regularization"
    UNCERTAINTY_ESTIMATION = "Uncertainty Estimation"
    IMPORTANCE_SAMPLING = "Importance Sampling"
    ONE_STEP = "One-step"
    MODEL_BASED = "Model-based"
    IMITATION_LEARNING = "Imitation Learning"
    TRAJECTORY_OPTIMIZATION = "Trajectory Optimization"

    def __str__(self) -> str:
        return self.value


class Method(Enum):
    BCQ = "BCQ"
    BEAR = "BEAR"
    BRAC_P = "BRACp"
    BRAC_V = "BRACv"
    FISHER_BRC = "FBRC"
    CQL = "CQL"
    AWR = "AWR"
    AWAC = "AWAC"
    COMBO = "COMBO"
    WIS = "WIS"
    DRIS = "DRIS"
    ALGAE_DICE = "AlgaeDICE"
    GEN_DICE = "GenDICE"
    DUAL_DICE = "DualDICE"
    ONESTEP_RL = "ORL"
    IQL = "IQL"
    MOPO = "MOPO"
    MOREL = "MOReL"
    BREMEN = "BREMEN"
    TD3BC = "TDBC"
    BAIL = "BAIL"
    CRR = "CRR"
    ABM_MPO = "ABMMPO"
    RVS_G = "RvSg"
    RVS_R = "RvSr"
    DT = "DT"
    TT = "TT"
    SAC = "SAC"
    SAC_OFFLINE = "SAC-off"
    BC = "BC"
    TEN_PERCENT_BC = "10%BC"


METHOD_NAME_TO_TAXONOMY_CLASSES: Dict[str, List[TaxonomyClass]] = {
    Method.SAC: [],
    Method.SAC_OFFLINE: [],
    Method.BC: [TaxonomyClass.IMITATION_LEARNING],
    Method.TEN_PERCENT_BC: [TaxonomyClass.IMITATION_LEARNING],
    Method.BCQ: [TaxonomyClass.DIRECT_POLICY_CONSTRAINT],
    Method.BEAR: [TaxonomyClass.IMPLICIT_POLICY_CONSTRAINT],
    Method.BRAC_P: [TaxonomyClass.DIRECT_POLICY_CONSTRAINT],
    Method.BRAC_V: [TaxonomyClass.IMPLICIT_POLICY_CONSTRAINT],
    Method.FISHER_BRC: [
        TaxonomyClass.DIRECT_POLICY_CONSTRAINT,
        TaxonomyClass.VALUE_REGULARIZATION,
    ],
    Method.CQL: [
        TaxonomyClass.POLICY_REGULARIZATION,
        TaxonomyClass.VALUE_REGULARIZATION,
    ],
    Method.AWR: [
        TaxonomyClass.IMPLICIT_POLICY_CONSTRAINT,
    ],
    Method.AWAC: [
        TaxonomyClass.IMPLICIT_POLICY_CONSTRAINT,
    ],
    Method.COMBO: [
        TaxonomyClass.POLICY_REGULARIZATION,
        TaxonomyClass.VALUE_REGULARIZATION,
        TaxonomyClass.MODEL_BASED,
    ],
    Method.WIS: [TaxonomyClass.IMPORTANCE_SAMPLING],
    Method.DRIS: [TaxonomyClass.IMPORTANCE_SAMPLING],
    Method.ALGAE_DICE: [
        TaxonomyClass.VALUE_REGULARIZATION,
        TaxonomyClass.IMPORTANCE_SAMPLING,
    ],
    Method.GEN_DICE: [TaxonomyClass.IMPORTANCE_SAMPLING],
    Method.DUAL_DICE: [TaxonomyClass.IMPORTANCE_SAMPLING],
    Method.ONESTEP_RL: [
        TaxonomyClass.IMPLICIT_POLICY_CONSTRAINT,
        TaxonomyClass.ONE_STEP,
    ],
    Method.IQL: [
        TaxonomyClass.IMPLICIT_POLICY_CONSTRAINT,
        TaxonomyClass.VALUE_REGULARIZATION,
        TaxonomyClass.ONE_STEP,
    ],
    Method.MOPO: [
        TaxonomyClass.UNCERTAINTY_ESTIMATION,
        TaxonomyClass.MODEL_BASED,
    ],
    Method.MOREL: [
        TaxonomyClass.UNCERTAINTY_ESTIMATION,
        TaxonomyClass.MODEL_BASED,
    ],
    Method.BREMEN: [
        TaxonomyClass.DIRECT_POLICY_CONSTRAINT,
        TaxonomyClass.MODEL_BASED,
    ],
    Method.TD3BC: [
        TaxonomyClass.IMPLICIT_POLICY_CONSTRAINT,
        TaxonomyClass.IMITATION_LEARNING,
    ],
    Method.BAIL: [
        TaxonomyClass.IMITATION_LEARNING,
    ],
    Method.CRR: [
        TaxonomyClass.IMPLICIT_POLICY_CONSTRAINT,
        TaxonomyClass.IMITATION_LEARNING,
    ],
    Method.ABM_MPO: [TaxonomyClass.IMITATION_LEARNING],
    Method.RVS_R: [TaxonomyClass.IMITATION_LEARNING],
    Method.RVS_G: [TaxonomyClass.IMITATION_LEARNING],
    Method.DT: [TaxonomyClass.TRAJECTORY_OPTIMIZATION],
    Method.TT: [TaxonomyClass.TRAJECTORY_OPTIMIZATION],
}


def invert_method_to_classes_dict() -> Dict[TaxonomyClass, List[Method]]:
    inverted_dict = defaultdict(list)
    for method, taxonomy_classes in METHOD_NAME_TO_TAXONOMY_CLASSES.items():
        for taxonomy_class in taxonomy_classes:
            inverted_dict[taxonomy_class].append(method)
    return inverted_dict


TAXONOMY_CLASS_TO_METHOD_NAMES: Dict[
    TaxonomyClass, List[Method]
] = invert_method_to_classes_dict()


DATASET_SUITE_TO_ENVIRONMENT_PREFIX: Dict[DatasetSuite, List[str]] = {
    DatasetSuite.MAZE2D: ["maze2d"],
    DatasetSuite.ANTMAZE: ["antmaze"],
    DatasetSuite.GYM_MUJOCO: ["halfcheetah", "hopper", "walker2d"],
    DatasetSuite.ADROIT: ["pen", "hammer", "door", "relocate"],
    DatasetSuite.KITCHEN: ["kitchen"],
    DatasetSuite.FLOW: ["flow"],
    DatasetSuite.CARLA: ["carla"],
}

DATASET_PROPERTY_TO_SUITES: Dict[DatasetProperty, List[DatasetSuite]] = {
    DatasetProperty.NARROW_AND_BIASED_DIST: [
        DatasetSuite.GYM_MUJOCO,
        DatasetSuite.ADROIT,
    ],
    DatasetProperty.UNDIRECTED_AND_MULTITASK_DATA: [
        DatasetSuite.MAZE2D,
        DatasetSuite.ANTMAZE,
        DatasetSuite.KITCHEN,
        DatasetSuite.CARLA,
    ],
    DatasetProperty.SPARSE_REWARDS: [DatasetSuite.ADROIT, DatasetSuite.ANTMAZE],
    DatasetProperty.SUBOPTIMAL_DATA: [DatasetSuite.GYM_MUJOCO],
    DatasetProperty.NON_REPRESENTABLE_BEHAVIOR_POLICY: [
        DatasetSuite.ADROIT,
        DatasetSuite.FLOW,
        DatasetSuite.CARLA,
    ],
    DatasetProperty.NON_MARKOVIAN_BEHAVIOR_POLICY: [
        DatasetSuite.MAZE2D,
        DatasetSuite.ANTMAZE,
    ],
    DatasetProperty.REALISTIC_DOMAIN: [
        DatasetSuite.ADROIT,
        DatasetSuite.KITCHEN,
        DatasetSuite.FLOW,
        DatasetSuite.CARLA,
    ],
}


def environment_names_for_suites(
    dataset_suites: List[DatasetSuite],
) -> Tuple[str]:
    env_names = []
    for dataset_suite in dataset_suites:
        env_names.extend(DATASET_SUITE_TO_ENVIRONMENT_PREFIX[dataset_suite])
    return tuple(env_names)


def environment_has_dataset_property(
    env_name: str, dataset_property: DatasetProperty
) -> bool:
    if dataset_property in DATASET_PROPERTY_TO_SUITES:
        return env_name.startswith(
            environment_names_for_suites(DATASET_PROPERTY_TO_SUITES[dataset_property])
        )
    return False


@lru_cache
def get_env_score_minmax() -> Dict[str, Tuple[float, float]]:
    env_scores: Dict[str, Tuple[float, float]] = defaultdict(
        lambda: (float("inf"), -float("inf"))
    )
    with open(SCORES_CSV, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                if key == "method" or value.startswith("--"):
                    continue
                value = float(value)
                cur_minmax = env_scores[key]
                env_scores[key] = (
                    min(value, cur_minmax[0]),
                    max(value, cur_minmax[1]),
                )
    return env_scores


def normalize_score(env_name: str, score: float) -> float:
    """Normalize scores between 0-100 for easier coloring afterwards."""
    env_minmax_scores = get_env_score_minmax()
    min_val, max_val = env_minmax_scores[env_name]
    return (score - min_val) / (max_val - min_val)


def dataset_property_scores_from_env_scores(
    env_scores: Dict[str, float]
) -> Dict[DatasetProperty, float]:
    dprop_scores: Dict[DatasetProperty, List[float]] = defaultdict(list)
    for env_name, score in env_scores.items():
        for dprop in DatasetProperty:
            if environment_has_dataset_property(env_name, dprop):
                dprop_scores[dprop].append(score)
    return dict_reduce(dprop_scores)


def dict_reduce(
    list_dict: Dict[Any, List[float]],
    reduce_fn: Callable[[np.ndarray], float] = np.mean,
) -> Dict[Any, float]:
    return {key: reduce_fn(values) for key, values in list_dict.items()}


def dataset_property_scores_from_algo_scores(
    algo_scores: Dict[str, Dict[DatasetProperty, float]]
) -> Dict[TaxonomyClass, Dict[DatasetProperty, float]]:
    taxonomy_classes_scores: Dict[
        str, Dict[DatasetProperty, List[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for algo, dprop_scores in algo_scores.items():
        method = algo_name_to_enum(algo)
        taxonomy_classes = METHOD_NAME_TO_TAXONOMY_CLASSES[method]
        for dprop, score in dprop_scores.items():
            for taxonomy_class in taxonomy_classes:
                taxonomy_classes_scores[taxonomy_class][dprop].append(score)

    return {
        taxonomy_class: dict_reduce(dprop_scores, reduce_fn=np.max)
        for taxonomy_class, dprop_scores in taxonomy_classes_scores.items()
    }


def rgba_to_tex_color(rgba_color: RGBAColor) -> str:
    rgba = tuple(map(lambda f: f"{f:.2f}", rgba_color))
    return f"rgb,1:red,{rgba[0]};green,{rgba[1]};blue,{rgba[2]}"


def mpl_to_tex_color(mpl_color: MatplotlibColor) -> str:
    if isinstance(mpl_color, str):
        if mpl_color == "w":
            return rgba_to_tex_color((1.0, 1.0, 1.0, 1.0))
        else:
            return rgba_to_tex_color((0.95, 0.95, 0.95, 1.0))
    return rgba_to_tex_color(mpl_color)


def draw_colorbar(size: int = 200) -> None:
    tex_colors = list(map(lambda i: rgba_to_tex_color(CMAP(i / size)), range(size)))
    with open("colorbar.csv", "w", encoding="utf8") as colorbar_file:
        colorbar_file.write("color\n")
        colorbar_file.write("\n".join(tex_colors))


def algo_name_to_enum(algo_name: str) -> Method:
    try:
        return Method(algo_name)
    except:
        raise ValueError(f"Algo {algo_name} doesn't have a corresponding enum.")


def draw_table(
    dprop_scores_per_key: Dict[str, Dict[DatasetProperty, float]],
    file_name: str = "dprop-scores.csv",
) -> None:
    columns = ["Method"]
    columns.extend(list(DatasetProperty))
    print(columns)
    colors, rows = [], []
    for key, dprop_scores in dprop_scores_per_key.items():
        color, row = ["w"], [str(key)]
        for dprop in columns[1:]:
            if dprop in dprop_scores:
                score = dprop_scores[dprop]
                color.append(CMAP(score))
            else:
                color.append("gray")
            row.append("")
        colors.append(color)
        rows.append(row)

    columns[1:] = list(map(lambda dprop: dprop.value, DatasetProperty))

    with open(file_name, "w", newline="", encoding="utf8") as csvfile:
        fieldnames = list(map(lambda s: s.lower(), columns))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()

        for method_row, color_row in zip(rows, colors):
            method_name = method_row[0]
            tex_color_row = list(map(mpl_to_tex_color, color_row[1:]))
            row = dict(zip(fieldnames, [method_name] + tex_color_row))
            writer.writerow(row)


def main() -> None:
    normalized_env_scores_per_algo: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(SCORES_CSV, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            algo = row["method"]
            for env_name in reader.fieldnames[1:]:
                raw_value = row[env_name]
                if raw_value.startswith("--"):
                    continue
                # Check why we always log zero even if item does not exist in collected environments.
                norm_value = normalize_score(env_name, float(raw_value))
                normalized_env_scores_per_algo[algo][env_name] = norm_value

    normalized_dprop_scores_per_algo = {
        algo: dataset_property_scores_from_env_scores(env_scores)
        for algo, env_scores in normalized_env_scores_per_algo.items()
    }

    normalized_dprop_scores_per_class = dataset_property_scores_from_algo_scores(
        normalized_dprop_scores_per_algo
    )

    draw_table(normalized_dprop_scores_per_algo, "dprop-scores-algo.csv")
    draw_table(normalized_dprop_scores_per_class, "dprop-scores-taxonomy.csv")
    draw_colorbar()


if __name__ == "__main__":
    main()
