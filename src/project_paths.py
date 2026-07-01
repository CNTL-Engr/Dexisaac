"""Path helpers anchored at the Dexisaac project root."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_MESH_ROOT = (PROJECT_ROOT / ".." / ".." / ".." / "mesh").resolve()


def project_path(*parts):
    """Return an absolute path under the Dexisaac project root."""
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_project_path(path):
    """Resolve a user path, treating relative paths as Dexisaac-root-relative."""
    if path is None:
        return None
    if path == "":
        return ""

    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    return str(PROJECT_ROOT / path_obj)


def external_mesh_path(*parts):
    """Return an absolute path under Dexisaac/../../../mesh."""
    return str(EXTERNAL_MESH_ROOT.joinpath(*parts))
