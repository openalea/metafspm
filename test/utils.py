import sys, importlib, pkgutil

def deep_reload_package(pkg_names: list):
    """
    Hard-reload a package and all its submodules.
    Returns the re-imported top-level package module object.
    """
    # 1) Drop the package and all children from sys.modules
    for pkg_name in pkg_names:
        to_drop = [m for m in list(sys.modules)
                if m == pkg_name or m.startswith(pkg_name + ".")]
        for m in to_drop:
            del sys.modules[m]
