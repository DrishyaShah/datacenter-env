# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Datacenter Env environment server components."""

try:
    from .client import DCEnv
    from .environment import DCEnvironment
    __all__ = ["DCEnvironment", "DCEnv"]
except ImportError:
    # openenv package not installed (training-only environment).
    # ClusterEnvironment imports work fine without DCEnv/DCEnvironment.
    __all__ = []
