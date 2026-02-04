# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections.abc import Sequence
from typing import Any

from lerobot.configs.types import PipelineFeatureType
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import DataProcessorPipeline, RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR
from collections import OrderedDict

_MAX_CACHE_SIZE = 512

_compile_cache: dict = OrderedDict()


def create_initial_features(
    action: RobotAction | None = None, observation: RobotObservation | None = None
) -> dict[PipelineFeatureType, dict[str, Any]]:
    """
    Creates the initial features dict for the dataset from action and observation specs.

    Args:
        action: A dictionary of action feature names to their types/shapes.
        observation: A dictionary of observation feature names to their types/shapes.

    Returns:
        The initial features dictionary structured by PipelineFeatureType.
    """
    features = {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}
    if action:
        features[PipelineFeatureType.ACTION] = action
    if observation:
        features[PipelineFeatureType.OBSERVATION] = observation
    return features


# Helper to filter state/action keys based on regex patterns.
def should_keep(key: str, patterns: tuple[str]) -> bool:
    if patterns is None:
        return True
    compiled = _compile_patterns(patterns)
    for pat in compiled:
        if pat.search(key):
            return True
    return False


def strip_prefix(key: str, prefixes_to_strip: tuple[str]) -> str:
    for prefix in prefixes_to_strip:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


# Define prefixes to strip from feature keys for clean names.
# Handles both fully qualified (e.g., "action.state") and short (e.g., "state") forms.
PREFIXES_TO_STRIP = tuple(
    f"{token}." for const in (ACTION, OBS_STATE, OBS_IMAGES) for token in (const, const.split(".")[-1])
)


def aggregate_pipeline_dataset_features(
    pipeline: DataProcessorPipeline,
    initial_features: dict[PipelineFeatureType, dict[str, Any]],
    *,
    use_videos: bool = True,
    patterns: Sequence[str] | None = None,
) -> dict[str, dict]:
    """
    Aggregates and filters pipeline features to create a dataset-ready features dictionary.

    This function transforms initial features using the pipeline, categorizes them as action or observations
    (image or state), filters them based on `use_videos` and `patterns`, and finally
    formats them for use with a Hugging Face LeRobot Dataset.

    Args:
        pipeline: The DataProcessorPipeline to apply.
        initial_features: A dictionary of raw feature specs for actions and observations.
        use_videos: If False, image features are excluded.
        patterns: A sequence of regex patterns to filter action and state features.
                  Image features are not affected by this filter.

    Returns:
        A dictionary of features formatted for a Hugging Face LeRobot Dataset.
    """
    all_features = pipeline.transform_features(initial_features)

    # Intermediate storage for categorized and filtered features.
    processed_features: dict[str, dict[str, Any]] = {
        ACTION: {},
        OBS_STR: {},
    }
    images_token = OBS_IMAGES.split(".")[-1]

    # Iterate through all features transformed by the pipeline.
    for ptype, feats in all_features.items():
        if ptype not in [PipelineFeatureType.ACTION, PipelineFeatureType.OBSERVATION]:
            continue

        for key, value in feats.items():
            # 1. Categorize the feature.
            is_action = ptype == PipelineFeatureType.ACTION
            # Observations are classified as images if their key matches image-related tokens or if the shape of the feature is 3.
            # All other observations are treated as state.
            is_image = not is_action and (
                (isinstance(value, tuple) and len(value) == 3)
                or (
                    key.startswith(f"{OBS_IMAGES}.")
                    or key.startswith(f"{images_token}.")
                    or f".{images_token}." in key
                )
            )

            # 2. Apply filtering rules.
            if is_image and not use_videos:
                continue
            if not is_image and not should_keep(key, patterns):
                continue

            # 3. Add the feature to the appropriate group with a clean name.
            name = strip_prefix(key, PREFIXES_TO_STRIP)
            if is_action:
                processed_features[ACTION][name] = value
            else:
                processed_features[OBS_STR][name] = value

    # Convert the processed features into the final dataset format.
    dataset_features = {}
    if processed_features[ACTION]:
        dataset_features.update(hw_to_dataset_features(processed_features[ACTION], ACTION, use_videos))
    if processed_features[OBS_STR]:
        dataset_features.update(hw_to_dataset_features(processed_features[OBS_STR], OBS_STR, use_videos))

    return dataset_features


def _compile_patterns(patterns):
    # Attempt to use the patterns tuple as a cache key; if it contains
    # unhashable elements, fall back to on-the-fly compilation to preserve
    # original exceptions and behavior.
    try:
        key = tuple(patterns)
    except TypeError:
        return tuple(
            pat if hasattr(pat, "search") and callable(pat.search) else re.compile(pat)
            for pat in patterns
        )

    cache = _compile_cache
    compiled = cache.get(key)
    if compiled is not None:
        # mark as recently used
        cache.move_to_end(key)
        return compiled

    compiled_list = []
    for pat in patterns:
        # If it's already a compiled pattern-like object, reuse it;
        # otherwise compile the pattern string.
        if hasattr(pat, "search") and callable(pat.search):
            compiled_list.append(pat)
        else:
            compiled_list.append(re.compile(pat))
    compiled = tuple(compiled_list)

    cache[key] = compiled
    if len(cache) > _MAX_CACHE_SIZE:
        cache.popitem(last=False)
    return compiled

def _compile_patterns(patterns):
    # Attempt to use the patterns tuple as a cache key; if it contains
    # unhashable elements, fall back to on-the-fly compilation to preserve
    # original exceptions and behavior.
    try:
        key = tuple(patterns)
    except TypeError:
        return tuple(
            pat if hasattr(pat, "search") and callable(pat.search) else re.compile(pat)
            for pat in patterns
        )

    cache = _compile_cache
    compiled = cache.get(key)
    if compiled is not None:
        # mark as recently used
        cache.move_to_end(key)
        return compiled

    compiled_list = []
    for pat in patterns:
        # If it's already a compiled pattern-like object, reuse it;
        # otherwise compile the pattern string.
        if hasattr(pat, "search") and callable(pat.search):
            compiled_list.append(pat)
        else:
            compiled_list.append(re.compile(pat))
    compiled = tuple(compiled_list)

    cache[key] = compiled
    if len(cache) > _MAX_CACHE_SIZE:
        cache.popitem(last=False)
    return compiled
