from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger
from pyVoIP.VoIP import VoIPCall

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    import pyVoIP
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use pyVoIP, you need to `pip install pyVoIP`.")
    raise Exception(f"Missing module: {e}")


class VoIPPhoneParams(TransportParams):
    add_wav_header: bool = False
    session_timeout: Optional[int] = None


class VoIPPhoneInputTransport(BaseInputTransport):
    def __init__(
        self,
        params: VoIPPhoneParams,
        call: VoIPCall,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._params = params
        self._call : VoIPCall = call
        self._server_task = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._server_task = self.create_task(self._server_task_handler())

        logger.info(f"NOW, answer call: {self._call}")
        self._call.answer()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._server_task:
            await self.cancel_task(self._server_task)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._server_task:
            await self.cancel_task(self._server_task)

    async def cleanup(self):
        await super().cleanup()
        if self._call:
            self._call.bye()
            self._call = None

    async def _server_task_handler(self):
        while True:
            in_data = self._call.read_audio()
            logger.debug(f"in_data size {len(in_data)}")
            frame = InputAudioRawFrame(
                audio=in_data,
                sample_rate=8000,
                num_channels=1,
            )
            await self.push_audio_frame(frame)


class VoIPPhoneOutputTransport(BaseOutputTransport):
    def __init__(self, params: VoIPPhoneParams, call: VoIPCall, **kwargs):
        super().__init__(params, **kwargs)
        self._params = params
        self._call = call

        # We only write audio frames from a single task, so only one thread
        # should be necessary.
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def start(self, frame: StartFrame):
        await super().start(frame)

    async def write_raw_audio_frames(self, frames: bytes):
        if self._call:
            logger.info(f"writing raw audio frames size: {len(frames)}")
            await self.get_event_loop().run_in_executor(
                self._executor, self._call.write_audio, frames
            )


class VoIPPhoneTransport(BaseTransport):
    def __init__(
        self,
        params: VoIPPhoneParams,
        voIPCall: VoIPCall
    ):
        super().__init__()
        self._params = params
        self._voIPCall = voIPCall

        self._input: Optional[VoIPPhoneInputTransport] = None
        self._output: Optional[VoIPPhoneOutputTransport] = None

    def input(self) -> VoIPPhoneInputTransport:
        if not self._input:
            self._input = VoIPPhoneInputTransport(self._params, self._voIPCall, name=self._input_name)
        return self._input

    def output(self) -> VoIPPhoneOutputTransport:
        if not self._output:
            self._output = VoIPPhoneOutputTransport(self._params, self._voIPCall, name=self._output_name)
        return self._output
