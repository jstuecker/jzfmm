from importlib.metadata import entry_points
import warnings

def discover_plugins():
    print("looking for plugins...")
    for ep in entry_points(group="fmdj.plugins"):
        print(f"Loading Plugin {ep.name}! qwe")
        try:
            obj = ep.load()
            if callable(obj):  # e.g., register()
                obj()
            # If ep points to a module (no ":register"), importing ran side effects.
        except Exception as e:
            warnings.warn(f"[fmdj] Plugin {ep.name} failed: {e}")