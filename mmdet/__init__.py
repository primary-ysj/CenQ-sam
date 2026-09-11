import mmcv
import numpy as np

from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


# This fork was originally based on MMDetection 2.13, but the 4090/CUDA 11.8
# environment uses mmcv-full 1.7.2. Keep the check explicit so incompatible
# mmcv 2.x installs still fail early.
mmcv_minimum_version = '1.3.2'
mmcv_maximum_version = '1.7.2'
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

__all__ = ['__version__', 'short_version']


def _patch_numpy_legacy_aliases():
    """Restore aliases removed in NumPy 1.24 for legacy project code."""
    aliases = {
        'bool': np.bool_,
        'int': int,
        'float': float,
        'object': object,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


_patch_numpy_legacy_aliases()
