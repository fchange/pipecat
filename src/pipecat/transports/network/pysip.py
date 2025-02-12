import asyncio
import io
import time
import typing
import wave
from typing import Awaitable, Callable

from PySIP.sip_account import SipCall
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams


class SipParams(TransportParams):
    session_timeout: int | None = None


class SipCallbacks(BaseModel):
    on_client_connected: Callable[[SipCall], Awaitable[None]]
    on_client_disconnected: Callable[[SipCall], Awaitable[None]]
    on_session_timeout: Callable[[SipCall], Awaitable[None]]


class SipInputTransport(BaseInputTransport):
    def __init__(
        self,
        sipCall: SipCall,
        params: SipParams,
        callbacks: SipCallbacks,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._sipCall = sipCall
        self._params = params
        self._callbacks = callbacks

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._params.serializer.setup(frame)
        if self._params.session_timeout:
            self._monitor_sipCall_task = self.create_task(self._monitor_sipCall())
        await self._callbacks.on_client_connected(self._sipCall)
        self._receive_task = self.create_task(self._receive_messages())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self.cancel_task(self._receive_task)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self.cancel_task(self._receive_task)

    def _iter_data(self) -> typing.AsyncIterator[bytes | str]:
        audio = self._sipCall.process_recorded_audio()
        yield audio

    async def _receive_messages(self):
        try:
            async for message in self._iter_data():
                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        await self._callbacks.on_client_disconnected(self._sipCall)

    async def _monitor_sipCall(self):
        """Wait for self._params.session_timeout seconds, if the sipCall is still open, trigger timeout event."""
        await asyncio.sleep(self._params.session_timeout)
        await self._callbacks.on_session_timeout(self._sipCall)


class SipOutputTransport(BaseOutputTransport):
    def __init__(self, sipCall: SipCall, params: SipParams, **kwargs):
        super().__init__(params, **kwargs)

        self._sipCall = sipCall
        self._params = params

        # write_raw_audio_frames() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._params.serializer.setup(frame)
        self._send_interval = (self._audio_chunk_size / self.sample_rate) / 2

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def write_raw_audio_frames(self, frames: bytes):
        frame = OutputAudioRawFrame(
            audio=frames,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                wav_frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                frame = wav_frame

        await self._write_frame(frame)

        self._sipCall_audio_buffer = bytes()

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

    async def _write_frame(self, frame: Frame):
        try:
            payload = await self._params.serializer.serialize(frame)
            if payload and self._sipCall.client_state == SipCallState.CONNECTED:
                await self._send_data(payload)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

    def _send_data(self, data: bytes):
        return self._sipCall.send_audio(data)

    async def _write_audio_sleep(self):
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class SipTransport(BaseTransport):
    def __init__(
        self,
        sipCall: SipCall,
        params: SipParams,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = SipCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        self._input = SipInputTransport(
            sipCall, self._params, self._callbacks, name=self._input_name
        )
        self._output = SipOutputTransport(
            sipCall, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")

    def input(self) -> SipInputTransport:
        return self._input

    def output(self) -> SipOutputTransport:
        return self._output

    async def _on_client_connected(self, sipCall):
        await self._call_event_handler("on_client_connected", sipCall)

    async def _on_client_disconnected(self, sipCall):
        await self._call_event_handler("on_client_disconnected", sipCall)

    async def _on_session_timeout(self, sipCall):
        await self._call_event_handler("on_session_timeout", sipCall)
