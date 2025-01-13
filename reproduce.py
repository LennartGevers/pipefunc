from dataclasses import dataclass
import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_outputs
from pipefunc._plotting import _all_type_annotations


@dataclass(frozen=True)
class Geometry:
    x: float
    y: float


@dataclass(frozen=True)
class Mesh:
    geometry: Geometry
    mesh_size: float


@dataclass(frozen=True)
class Materials:
    geometry: Geometry
    materials: list[str]


@dataclass(frozen=True)
class Electrostatics:
    mesh: Mesh
    materials: Materials
    voltages: list[float]



@pipefunc(output_name="geo")
def make_geometry(x: float, y: float) -> Geometry:
    return Geometry(x, y)


@pipefunc(output_name=("mesh", "coarse_mesh"))
def make_mesh(
    geo: Geometry,
    mesh_size: float,
    coarse_mesh_size: float,
) -> tuple[Mesh, Mesh]:
    return Mesh(geo, mesh_size), Mesh(geo, coarse_mesh_size)


@pipefunc(output_name="materials")
def make_materials(geo: Geometry) -> Materials:
    return Materials(geo, ["i", "j", "c"])


@pipefunc(output_name="electrostatics", mapspec="V_left[i], V_right[j] -> electrostatics[i, j]")
def run_electrostatics(
    mesh: Mesh,
    materials: Materials,
    V_left: float,  # noqa: N803
    V_right: float,  # noqa: N803
) -> Electrostatics:
    return Electrostatics(mesh, materials, [V_left, V_right])


@pipefunc(output_name="charge", mapspec="electrostatics[i, j] -> charge[i, j]")
def get_charge(electrostatics: Electrostatics) -> float:
    # obviously not actually the charge; but we should return _some_ number that
    # is "derived" from the electrostatics.
    return sum(electrostatics.voltages)


# No mapspec: function receives the full 2D array of charges!
@pipefunc(output_name="average_charge")
def average_charge(charge: np.ndarray) -> float:
    return np.mean(charge)


pipeline_charge = Pipeline(
    [make_geometry, make_mesh, make_materials, run_electrostatics, get_charge, average_charge],
)

# Add a cross-product of x and y
pipeline_charge.add_mapspec_axis("x", axis="a")
pipeline_charge.add_mapspec_axis("y", axis="b")

# And also a cross-product of the zipped mesh_size and coarse_mesh_size
pipeline_charge.add_mapspec_axis("mesh_size", axis="c")
pipeline_charge.add_mapspec_axis("coarse_mesh_size", axis="c")

pipeline_charge2 = pipeline_charge.copy()
nested_func = pipeline_charge2.nest_funcs(
    {"electrostatics", "charge"},
    new_output_name="charge",
    # We can also specify `("charge", "electrostatics")` to get both outputs
)

pipeline_charge2.visualize(orient="TB")
print("all_type_annotations", _all_type_annotations(nested_func.pipeline.graph))