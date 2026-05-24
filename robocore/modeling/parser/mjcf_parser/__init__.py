# Copyright 2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""PyMJCF: an MJCF object-model library."""

from importlib import import_module

_LAZY_EXPORTS = {
    'from_path': ('robocore.modeling.parser.mjcf_parser.parser', 'from_path'),
    'MJCFParser': ('robocore.modeling.parser.mjcf_parser.wrapper', 'MJCFParser'),
    'load_mjcf': ('robocore.modeling.parser.mjcf_parser.wrapper', 'load_mjcf'),
    'MJCF': ('robocore.modeling.parser.mjcf_parser.mjcf', 'MJCF'),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
