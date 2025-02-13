import gzip
import json
import uuid
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import TTSService
from pipecat.services.websocket_service import WebsocketService

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use huoshan, you need to `pip install pipecat-ai[huoshan]`. Also, set `HUOSHAN_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

default_header = bytearray(b'\x11\x10\x11\x00')

class HuoShanAudioTTSService(TTSService, WebsocketService):
    def __init__(
            self,
            appid: str,
            api_token: str,
            cluster: str,
            voice_type: str ="BV700_V2_streaming",
            sample_rate: int = 8000,
            **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._request_id = None
        self._voice_type = voice_type
        self._appid = appid
        self._api_token = api_token
        self._cluster = cluster
        self._host = "openspeech.bytedance.com"
        self._base_url = f"wss://{self._host}/api/v1/tts/ws_binary"

        self._sample_rate = sample_rate

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await self._connect_websocket()
        self._receive_task = self.create_task(self._receive_task_handler(self.push_error))

    async def _disconnect(self):
        await self._disconnect_websocket()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def _connect_websocket(self):
        try:
            logger.debug("Connecting to HuoShan Audio, api_token=" + self._api_token)
            headers = {"Authorization": f"Bearer; {self._api_token}"}
            self._websocket = await websockets.connect(self._base_url, extra_headers=headers)
        except Exception as e:
            logger.error(f"HuoShan Audio initialization error: {e}")
            self._websocket = None

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from HuoShan Audio")
                await self._websocket.close()
                self._websocket = None
            self._request_id = None
            self._started = False
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        while True:
            res = await self._get_websocket().recv()
            done = await self.parse_response(res)
            if done:
                await self.push_frame(TTSStoppedFrame())
                break

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSSpeakFrame):
            await self.pause_processing_frames()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._request_id:
            await self.pause_processing_frames()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.resume_processing_frames()

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating HuoShan TTS: [{text}]")
        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()

            if not self._request_id:
                await self.start_ttfb_metrics()
                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()
                self._request_id = str(uuid.uuid4())

            try:
                await self._get_websocket().send(self.generate_request(text))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()

            yield None

        except Exception as e:
            logger.error(f"Error generating TTS: {e}")
            yield ErrorFrame(f"Error: {str(e)}")

    def generate_request(self, text) -> bytearray:
        submit_request_json = {
            "app": {
                "appid": self._appid,
                "token": self._api_token,
                "cluster": self._cluster
            },
            "user": {
                "uid": self._request_id
            },
            "audio": {
                "voice_type": self._voice_type,
                "encoding": "pcm",
                "rate": self._sample_rate,
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
                "emotion": "happy",
                "language": "cn"
            },
            "request": {
                "reqid": self._request_id,
                "text": text,
                "text_type": "plain",
                "operation": "submit"
            }
        }
        payload_bytes = str.encode(json.dumps(submit_request_json))
        payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
        full_client_request = bytearray(default_header)
        full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
        full_client_request.extend(payload_bytes)  # payload
        return full_client_request

    async def parse_response(self, res):
        protocol_version = res[0] >> 4
        header_size = res[0] & 0x0f
        message_type = res[1] >> 4
        message_type_specific_flags = res[1] & 0x0f
        serialization_method = res[2] >> 4
        message_compression = res[2] & 0x0f
        reserved = res[3]
        header_extensions = res[4:header_size * 4]
        payload = res[header_size * 4:]

        if message_type == 0xb:  # audio-only server response
            if message_type_specific_flags == 0:  # no sequence number as ACK
                return False
            else:
                sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                payload = payload[8:]

            frame = TTSAudioRawFrame(payload, self.sample_rate, 1)
            await self.push_frame(frame)
            await self.stop_ttfb_metrics()

            if sequence_number < 0:
                return True
            else:
                return False
        elif message_type == 0xf:
            code = int.from_bytes(payload[:4], "big", signed=False)
            msg_size = int.from_bytes(payload[4:8], "big", signed=False)
            error_msg = payload[8:]
            if message_compression == 1:
                error_msg = gzip.decompress(error_msg)
            error_msg = str(error_msg, "utf-8")
            logger.error(f"Error message code: {code}, size: {msg_size} bytes, message: {error_msg}")
            return True
        elif message_type == 0xc:
            msg_size = int.from_bytes(payload[:4], "big", signed=False)
            payload = payload[4:]
            if message_compression == 1:
                payload = gzip.decompress(payload)
            logger.error(f"Frontend message: {payload}")
        else:
            logger.error("Undefined message type!")
            return True
