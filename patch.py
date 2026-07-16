import sys
import types
if 'mmcv._ext' not in sys.modules:
    sys.modules['mmcv._ext'] = types.ModuleType('mmcv._ext')
    sys.modules['mmcv._ext'].active_rotated_filter_forward = lambda *args: None
    sys.modules['mmcv._ext'].active_rotated_filter_backward = lambda *args: None
