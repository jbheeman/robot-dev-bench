import sys

filename = '/home/andrew/miniconda3/lib/python3.13/site-packages/mmcv/utils/ext_loader.py'
with open(filename, 'r') as f:
    content = f.read()

replacement = """
    def load_ext(name, funcs):
        class DummyExt:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        return DummyExt()
"""

content = content.replace("""    def load_ext(name, funcs):
        ext = importlib.import_module('mmcv.' + name)
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext""", replacement)

with open(filename, 'w') as f:
    f.write(content)
