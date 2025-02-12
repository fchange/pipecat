#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements SenseVoice transcription with a locally-downloaded model."""

import asyncio
from typing import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.ai_services import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

try:
    from funasr_onnx import Paraformer
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use SenseVoice, you need to `pip install pipecat-ai[SenseVoice]`.")
    raise Exception(f"Missing module: {e}")

class SenseVoiceSTTService(SegmentedSTTService):
    """Class to transcribe audio with a locally-downloaded SenseVoice model"""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._load()

    def can_generate_metrics(self) -> bool:
        return True

    def _load(self):
        """Loads the SenseVoice model. Note that if this is the first time
        this model is being run, it will take time to download.
        """
        logger.debug("Loading SenseVoice model...")

        model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        model = Paraformer(model_dir, batch_size=1, quantize=True)

        logger.debug("Loaded SenseVoice model")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes given audio using SenseVoice"""
        if not self._model:
            logger.error(f"{self} error: SenseVoice model not available")
            yield ErrorFrame("SenseVoice model not available")
            return

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = await asyncio.to_thread(self.model, audio_float)
        text: str = ""
        for segment in segments:
            if segment.no_speech_prob < self._no_speech_prob:
                text += f"{segment.text} "

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, "", time_now_iso8601())
