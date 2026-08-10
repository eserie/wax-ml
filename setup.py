# Copyright 2021 The WAX-ML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build-time shim: keep the test suite out of the distributions.

The whole build configuration lives in ``pyproject.toml``; this file exists for one
reason. WAX-ML colocates its tests with the code they exercise (``stream.py`` next to
``stream_test.py``), which setuptools has no declarative way to exclude -- ``packages.find``
excludes whole packages, and ``exclude-package-data`` does not apply to modules. Without
this hook every ``*_test.py`` is copied verbatim into the sdist and the wheel: 58 files
and roughly 400 kB that no installed user can run and that nothing imports.

Overriding ``find_package_modules`` filters them out of both artifacts at once, because
setuptools derives the sdist file list from the same call. The alternative -- moving the
tests to a top-level ``tests/`` tree -- is the cleaner layout but a far larger diff, and it
would break the ``pytest src/wax/<module>_test.py`` invocations documented in CLAUDE.md
and used by CI.
"""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):  # noqa: N801 -- distutils command classes are lowercase.
    """``build_py`` that drops ``*_test.py`` from the collected modules."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, module, path) for pkg, module, path in modules if not module.endswith("_test")
        ]


setup(cmdclass={"build_py": build_py})
