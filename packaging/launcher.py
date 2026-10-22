"""PyInstaller's entry script.

``app/main.py`` uses relative imports, so it cannot be the script itself: run as
a file it is ``__main__`` with no package, and the first ``from .core`` fails.
"""

import sys

from yolo_assistant.app.main import main

sys.exit(main())
