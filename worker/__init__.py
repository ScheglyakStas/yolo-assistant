"""Worker package: runs inside the managed venv, never imports PySide6.

Shipped as loose source rather than inside the PyInstaller archive, because the
venv's interpreter has to import it and cannot read ``sys._MEIPASS``.
"""
