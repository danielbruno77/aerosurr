"""Write an OpenFOAM v2506 case (snappyHexMesh + simpleFoam, k-omega SST) for one geometry.

The dictionaries follow the v2506 `incompressible/simpleFoam/motorBike` tutorial. Boundary
conditions follow the DrivAerML HRLES setup where it matters (see configs/rans/*.yaml).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from aerosurr.config import Freestream, RansConfig, RansMesh
from aerosurr.physics.forces import Reference

CAR = "car"  # the single wall patch for the whole vehicle
SURFACE_FO = "carSurface"  # postProcessing/carSurface/<iter>/car.vtp


@dataclass(frozen=True)
class BlockGrid:
    """Background mesh: two blocks split at x_bl, cubic cells of size h."""

    h: float
    x: tuple[float, float, float]  # x0, x_bl, x2
    y: tuple[float, float]
    z: tuple[float, float]
    n: tuple[int, int, int, int]  # cells: block A in x, block B in x, y, z

    @property
    def n_cells(self) -> int:
        na, nb, ny, nz = self.n
        return (na + nb) * ny * nz


def block_grid(cfg: RansConfig, mesh: RansMesh) -> BlockGrid:
    """Snap the tunnel extents to whole cells so every background cell is a cube."""
    d, h = cfg.domain, mesh.base_cell
    na = max(1, math.ceil((d.x_bl - d.x_min) / h))
    nb = max(1, math.ceil((d.x_max - d.x_bl) / h))
    ny = 2 * max(1, math.ceil(d.y_half / h))
    nz = max(1, math.ceil(d.height / h))
    return BlockGrid(
        h=h,
        x=(d.x_bl - na * h, d.x_bl, d.x_bl + nb * h),
        y=(-ny * h / 2, ny * h / 2),
        z=(d.ground_z, d.ground_z + nz * h),
        n=(na, nb, ny, nz),
    )


def _header(cls: str, obj: str, location: str | None = None) -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n"
    )


def _v(p: tuple[float, ...]) -> str:
    return "(" + " ".join(f"{x:.6g}" for x in p) + ")"


def block_mesh_dict(g: BlockGrid) -> str:
    x0, x1, x2 = g.x
    (y0, y1), (z0, z1) = g.y, g.z
    verts: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}

    def vid(p: tuple[float, float, float]) -> int:
        if p not in index:
            index[p] = len(verts)
            verts.append(p)
        return index[p]

    def hexa(xa: float, xb: float) -> list[int]:
        return [
            vid(p)
            for p in [
                (xa, y0, z0), (xb, y0, z0), (xb, y1, z0), (xa, y1, z0),
                (xa, y0, z1), (xb, y0, z1), (xb, y1, z1), (xa, y1, z1),
            ]
        ]  # fmt: skip

    a, b = hexa(x0, x1), hexa(x1, x2)

    def face(hx: list[int], which: str) -> str:
        idx = {"xmin": (0, 4, 7, 3), "xmax": (1, 2, 6, 5), "ymin": (0, 1, 5, 4),
               "ymax": (3, 7, 6, 2), "zmin": (0, 3, 2, 1), "zmax": (4, 5, 6, 7)}[which]  # fmt: skip
        return "(" + " ".join(str(hx[i]) for i in idx) + ")"

    patches = {
        "inlet": ("patch", [face(a, "xmin")]),
        "outlet": ("patch", [face(b, "xmax")]),
        "sides": ("symmetry", [face(a, "ymin"), face(a, "ymax"), face(b, "ymin"), face(b, "ymax")]),
        "top": ("symmetry", [face(a, "zmax"), face(b, "zmax")]),
        "groundSlip": ("symmetryPlane", [face(a, "zmin")]),
        "ground": ("wall", [face(b, "zmin")]),
    }
    na, nb, ny, nz = g.n
    out = [_header("dictionary", "blockMeshDict"), "scale 1;\n\nvertices\n(\n"]
    out += [f"    {_v(p)}\n" for p in verts]
    out.append(");\n\nblocks\n(\n")
    out.append(f"    hex ({' '.join(map(str, a))}) ({na} {ny} {nz}) simpleGrading (1 1 1)\n")
    out.append(f"    hex ({' '.join(map(str, b))}) ({nb} {ny} {nz}) simpleGrading (1 1 1)\n")
    out.append(");\n\nedges ();\n\nboundary\n(\n")
    for name, (kind, faces) in patches.items():
        out.append(f"    {name}\n    {{\n        type {kind};\n        faces\n        (\n")
        out += [f"            {f}\n" for f in faces]
        out.append("        );\n    }\n")
    out.append(");\n\nmergePatchPairs ();\n")
    return "".join(out)


def snappy_dict(cfg: RansConfig, mesh: RansMesh) -> str:
    boxes_geo = "".join(
        f"    {b.name}\n    {{\n        type box;\n        min {_v(b.min)};\n"
        f"        max {_v(b.max)};\n    }}\n"
        for b in mesh.boxes
    )
    boxes_ref = "".join(
        f"        {b.name}\n        {{\n            mode inside;\n"
        f"            levels ((1e15 {b.level}));\n        }}\n"
        for b in mesh.boxes
    )
    lo, hi = mesh.surface_levels
    return (
        _header("dictionary", "snappyHexMeshDict")
        + f"""castellatedMesh true;
snap            true;
addLayers       false;   // coarse LF: wall functions on the snapped cells

geometry
{{
    {CAR}.stl
    {{
        type triSurfaceMesh;
        name {CAR};
    }}
{boxes_geo}}}

castellatedMeshControls
{{
    maxLocalCells {mesh.max_local_cells};
    maxGlobalCells {mesh.max_global_cells};
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels {mesh.n_cells_between_levels};
    features
    (
        {{
            file "{CAR}.eMesh";
            level {mesh.feature_level};
        }}
    );
    refinementSurfaces
    {{
        {CAR}
        {{
            level ({lo} {hi});
            patchInfo
            {{
                type wall;
            }}
        }}
    }}
    resolveFeatureAngle {mesh.resolve_feature_angle:g};
    refinementRegions
    {{
{boxes_ref}    }}
    locationInMesh {_v(cfg.domain.location_in_mesh)};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{}}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    #includeEtc "caseDicts/meshQualityDict"
    nSmoothScale 4;
    errorReduction 0.75;
}}

mergeTolerance 1e-6;
"""
    )


def feature_dict(mesh: RansMesh) -> str:
    return _header("dictionary", "surfaceFeatureExtractDict") + (
        f"{CAR}.stl\n{{\n    extractionMethod    extractFromSurface;\n"
        f"    includedAngle       {mesh.feature_angle:g};\n"
        "    subsetFeatures\n    {\n        nonManifoldEdges no;\n        openEdges yes;\n    }\n"
        "    writeObj            no;\n}\n"
    )


def control_dict(cfg: RansConfig, fs: Freestream, ref: Reference) -> str:
    it = cfg.solver.iterations
    return (
        _header("dictionary", "controlDict")
        + f"""application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {it};
deltaT          1;
writeControl    timeStep;
writeInterval   {it};
purgeWrite      0;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;

functions
{{
    forceCoeffs1
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        log             no;
        patches         ({CAR});
        rho             rhoInf;
        rhoInf          {fs.rho_inf:g};
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        CofR            {_v(ref.cor)};
        pitchAxis       (0 1 0);
        magUInf         {fs.u_inf:g};
        lRef            {ref.l_ref:g};
        Aref            {ref.a_ref:g};
    }}
    // At the final write: wall shear stress and y+ on the car, then the car surface with
    // p and wallShearStress as face data (merged across ranks) for LF->HF mapping.
    wallShearStress
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        patches         ({CAR});
    }}
    yPlus
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
    }}
    {SURFACE_FO}
    {{
        type            surfaces;
        libs            (sampling);
        writeControl    writeTime;
        surfaceFormat   vtk;
        formatOptions   {{ vtk {{ format binary; }} }}
        fields          (p wallShearStress yPlus);
        surfaces
        {{
            {CAR}
            {{
                type        patch;
                patches     ({CAR});
                interpolate false;
            }}
        }}
    }}
}}
"""
    )


FV_SCHEMES = (
    _header("dictionary", "fvSchemes")
    + """ddtSchemes      { default steadyState; }
gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    turbulence      bounded Gauss upwind;
    div(phi,k)      $turbulence;
    div(phi,omega)  $turbulence;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.5; }
wallDist        { method meshWave; }
"""
)

FV_SOLUTION = (
    _header("dictionary", "fvSolution")
    + """solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }
    Phi { $p; }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
}
potentialFlow   { nNonOrthogonalCorrectors 10; }
relaxationFactors
{
    equations
    {
        U               0.9;
        k               0.7;
        omega           0.7;
    }
}
cache { grad(U); }
"""
)


def _field(cls: str, name: str, dims: str, internal: str, patches: dict[str, str]) -> str:
    body = "".join(f"    {p}\n    {{\n{v}    }}\n" for p, v in patches.items())
    return (
        _header(cls, name, "0")
        + f"dimensions      {dims};\ninternalField   {internal};\n\nboundaryField\n{{\n"
        + '    #includeEtc "caseDicts/setConstraintTypes"\n'
        + body
        + "}\n"
    )


def initial_fields(cfg: RansConfig, fs: Freestream) -> dict[str, str]:
    """0/ fields. symmetry, symmetryPlane and processor patches come from setConstraintTypes."""
    s = cfg.solver
    k = 1.5 * (s.turbulence_intensity * fs.u_inf) ** 2
    omega = k / (fs.nu * s.viscosity_ratio)
    u = f"uniform ({fs.u_inf:g} 0 0)"
    walls = ("ground", CAR)
    fixed = "        type fixedValue;\n        value $internalField;\n"
    zg = "        type zeroGradient;\n"

    def io(inlet_value: str) -> str:
        return (
            f"        type inletOutlet;\n        inletValue {inlet_value};\n"
            "        value $internalField;\n"
        )

    def with_walls(base: dict[str, str], wall: str) -> dict[str, str]:
        return base | {w: wall for w in walls}

    wall_k = "        type kqRWallFunction;\n        value $internalField;\n"
    wall_omega = "        type omegaWallFunction;\n        value $internalField;\n"
    wall_nut = "        type nutkWallFunction;\n        value uniform 0;\n"
    calc = "        type calculated;\n        value uniform 0;\n"
    vec, sca = "volVectorField", "volScalarField"
    return {
        "U": _field(vec, "U", "[0 1 -1 0 0 0 0]", u,
                    with_walls({"inlet": fixed, "outlet": io("uniform (0 0 0)")},
                               "        type noSlip;\n")),
        "p": _field(sca, "p", "[0 2 -2 0 0 0 0]", f"uniform {fs.p_inf:g}",
                    with_walls({"inlet": zg, "outlet": fixed}, zg)),
        "k": _field(sca, "k", "[0 2 -2 0 0 0 0]", f"uniform {k:.6g}",
                    with_walls({"inlet": fixed, "outlet": io("$internalField")}, wall_k)),
        "omega": _field(sca, "omega", "[0 0 -1 0 0 0 0]", f"uniform {omega:.6g}",
                        with_walls({"inlet": fixed, "outlet": io("$internalField")}, wall_omega)),
        "nut": _field(sca, "nut", "[0 2 -1 0 0 0 0]", "uniform 0",
                      with_walls({"inlet": calc, "outlet": calc}, wall_nut)),
    }  # fmt: skip


def write_case(
    case: Path,
    cfg: RansConfig,
    preset: str,
    stl: Path,
    fs: Freestream,
    ref: Reference,
    n_ranks: int,
) -> BlockGrid:
    """Create `case/` with system/, constant/ and 0.orig/. The STL becomes one `car` solid."""
    from aerosurr.rans.stl import read_stl, write_binary_stl

    if preset not in cfg.mesh_presets:
        raise KeyError(f"mesh preset {preset!r} not in {sorted(cfg.mesh_presets)}")
    mesh = cfg.mesh_presets[preset]
    grid = block_grid(cfg, mesh)
    surf = read_stl(stl)
    lo = surf.bounds[0]
    if lo[2] > cfg.domain.ground_z:
        raise ValueError(f"STL z_min {lo[2]:.4f} is above the ground {cfg.domain.ground_z}")
    for d in ("system", "constant/triSurface", "0.orig"):
        (case / d).mkdir(parents=True, exist_ok=True)
    write_binary_stl(case / "constant" / "triSurface" / f"{CAR}.stl", surf.triangles, CAR)
    files = {
        "system/blockMeshDict": block_mesh_dict(grid),
        "system/snappyHexMeshDict": snappy_dict(cfg, mesh),
        "system/surfaceFeatureExtractDict": feature_dict(mesh),
        "system/controlDict": control_dict(cfg, fs, ref),
        "system/fvSchemes": FV_SCHEMES,
        "system/fvSolution": FV_SOLUTION,
        "system/decomposeParDict": _header("dictionary", "decomposeParDict")
        + f"numberOfSubdomains {n_ranks};\nmethod          scotch;\n",
        "constant/transportProperties": _header("dictionary", "transportProperties")
        + f"transportModel  Newtonian;\nnu              {fs.nu:g};\n",
        "constant/turbulenceProperties": _header("dictionary", "turbulenceProperties")
        + "simulationType  RAS;\nRAS\n{\n    RASModel        kOmegaSST;\n"
        "    turbulence      on;\n    printCoeffs     on;\n}\n",
    }
    for name, text in initial_fields(cfg, fs).items():
        files[f"0.orig/{name}"] = text
    for rel, text in files.items():
        (case / rel).write_text(text)
    return grid
