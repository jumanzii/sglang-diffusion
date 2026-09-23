# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class QwenImage21SamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024
    num_frames: int = 1
    guidance_scale: float = 1.0
    num_inference_steps: int = 40
    negative_prompt: str | None = None
    # Lossy opt-in: a calibrated DPCache schedule artifact (the parsed JSON
    # object). Steps outside its full_steps skip every transformer block and
    # extrapolate the final block feature; see runtime/cache/dpcache.py.
    dpcache_schedule: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dpcache_schedule is not None:
            from sglang.multimodal_gen.runtime.cache.dpcache import validate_schedule

            validate_schedule(self.dpcache_schedule)

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset({"dpcache_schedule"})
