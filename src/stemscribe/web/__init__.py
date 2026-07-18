"""Local web UI for stemscribe. See server.main().

The module is `server`, not `app`: exporting an `app` attribute here would
shadow an `app` submodule of the same name, so `import stemscribe.web.app`
would hand back the FastAPI instance instead of the module.
"""
from .server import app, main

__all__ = ["app", "main"]
