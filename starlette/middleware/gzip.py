from __future__ import annotations

import math
import zlib
from typing import ClassVar, NoReturn

import anyio.lowlevel
import anyio.to_thread

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# TODO(v2): We should rename `DEFAULT_EXCLUDED_CONTENT_TYPES` to `DEFAULT_EXCLUDE_CONTENT_TYPES`.
DEFAULT_EXCLUDED_CONTENT_TYPES = (
    "application/gzip",
    "application/x-gzip",
    "application/zip",
    "audio/*",
    "font/woff",
    "font/woff2",
    "image/avif",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
    "text/event-stream",
    "video/*",
)

_compression_capacity_limiter: anyio.lowlevel.RunVar[anyio.CapacityLimiter] = anyio.lowlevel.RunVar(
    "_compression_capacity_limiter"
)

_SUPPORTED_CONTENT_ENCODINGS = ("gzip", "deflate", "identity")


def _get_compression_capacity_limiter() -> anyio.CapacityLimiter:
    """Return the capacity limiter used for worker-thread compression."""
    try:
        return _compression_capacity_limiter.get()
    except LookupError:
        limiter = anyio.CapacityLimiter(40)
        _compression_capacity_limiter.set(limiter)
        return limiter


def _select_content_encoding(accept_encoding: str) -> str | None:
    accepted_qualities: dict[str, float] = {}
    wildcard_quality: float | None = None

    for field in accept_encoding.split(","):
        coding, separator, parameters = field.partition(";")
        coding = coding.strip().lower()
        if not coding:
            continue

        quality = 1.0
        if separator:
            valid = True
            for parameter in parameters.split(";"):
                name, _has_value, value = parameter.partition("=")
                if name.strip().lower() != "q":
                    continue
                try:
                    quality = float(value.strip())
                except ValueError:
                    valid = False
                    break
                if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
                    valid = False
                    break
            if not valid:
                continue

        if coding == "*":
            if wildcard_quality is None:
                wildcard_quality = quality
        elif coding in _SUPPORTED_CONTENT_ENCODINGS and coding not in accepted_qualities:
            accepted_qualities[coding] = quality

    candidates: list[tuple[float, int, int, str]] = []
    for preference, encoding in enumerate(_SUPPORTED_CONTENT_ENCODINGS):
        selected_quality = accepted_qualities.get(encoding)
        if selected_quality is None and wildcard_quality is not None:
            selected_quality = wildcard_quality
        if selected_quality is None and encoding == "identity":
            selected_quality = 1.0
        if selected_quality is None or selected_quality == 0.0:
            continue

        specificity = 1 if encoding in accepted_qualities else 0
        candidates.append((selected_quality, specificity, -preference, encoding))

    if not candidates:
        return None
    return max(candidates)[-1]


class GZipMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int = 500,
        compresslevel: int = 9,
        thread_minimum_size: int = 128 * 1024,  # 128 KiB
        *,
        exclude_content_types: tuple[str, ...] = DEFAULT_EXCLUDED_CONTENT_TYPES,
    ) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel
        self.thread_minimum_size = thread_minimum_size
        self.exclude_content_types = _normalize_content_types(exclude_content_types)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # pragma: no cover
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_encoding = _select_content_encoding(headers.get("Accept-Encoding", ""))
        responder: ASGIApp
        if content_encoding in {"gzip", "deflate"}:
            responder_class = GZipResponder if content_encoding == "gzip" else DeflateResponder
            responder = responder_class(
                self.app,
                self.minimum_size,
                compresslevel=self.compresslevel,
                thread_minimum_size=self.thread_minimum_size,
                exclude_content_types=self.exclude_content_types,
            )
        else:
            responder = IdentityResponder(self.app, self.minimum_size, exclude_content_types=self.exclude_content_types)

        await responder(scope, receive, send)


class IdentityResponder:
    content_encoding: str

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int,
        *,
        exclude_content_types: tuple[str, ...] = DEFAULT_EXCLUDED_CONTENT_TYPES,
    ) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.exclude_content_types = _normalize_content_types(exclude_content_types)
        self.send: Send = unattached_send
        self.initial_message: Message = {}
        self.started = False
        self.content_encoding_set = False
        self.content_type_is_excluded = False
        self.partial_response = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.send = send
        await self.app(scope, receive, self.send_with_compression)

    async def send_with_compression(self, message: Message) -> None:
        message_type = message["type"]
        if message_type == "http.response.start":
            # Don't send the initial message until we've determined how to
            # modify the outgoing headers correctly.
            self.initial_message = message
            headers = Headers(raw=self.initial_message["headers"])
            self.content_encoding_set = "content-encoding" in headers
            self.partial_response = message["status"] == 206
            media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
            media_types = {media_type, media_type.partition("/")[0] + "/*"}
            self.content_type_is_excluded = not media_types.isdisjoint(self.exclude_content_types)
        elif message_type == "http.response.body" and (
            self.content_encoding_set or self.partial_response or self.content_type_is_excluded
        ):
            if not self.started:
                self.started = True
                await self.send(self.initial_message)
            await self.send(message)
        elif message_type == "http.response.body" and not self.started:
            self.started = True
            body = message.get("body", b"")
            more_body = message.get("more_body", False)
            if len(body) < self.minimum_size and not more_body:
                # Don't apply compression to small outgoing responses.
                await self.send(self.initial_message)
                await self.send(message)
            elif not more_body:
                # Standard response.
                body = await self.apply_compression(body, more_body=False)

                headers = MutableHeaders(raw=self.initial_message["headers"])
                headers.add_vary_header("Accept-Encoding")
                if body != message["body"]:
                    headers["Content-Encoding"] = self.content_encoding
                    headers["Content-Length"] = str(len(body))
                    message["body"] = body

                await self.send(self.initial_message)
                await self.send(message)
            else:
                # Initial body in streaming response.
                body = await self.apply_compression(body, more_body=True)

                headers = MutableHeaders(raw=self.initial_message["headers"])
                headers.add_vary_header("Accept-Encoding")
                if body != message["body"]:
                    headers["Content-Encoding"] = self.content_encoding
                    del headers["Content-Length"]
                    message["body"] = body

                await self.send(self.initial_message)
                await self.send(message)
        elif message_type == "http.response.body":
            # Remaining body in streaming response.
            body = message.get("body", b"")
            more_body = message.get("more_body", False)

            message["body"] = await self.apply_compression(body, more_body=more_body)

            await self.send(message)
        elif message_type == "http.response.pathsend":  # pragma: no branch
            # Don't apply GZip to pathsend responses
            await self.send(self.initial_message)
            await self.send(message)

    async def apply_compression(self, body: bytes, *, more_body: bool) -> bytes:
        """Apply compression on the response body.

        If more_body is False, the compression stream is finalized. Compression
        resources are only allocated once a body is actually compressed.
        """
        return body


class CompressedResponder(IdentityResponder):
    content_encoding: str
    wbits: ClassVar[int]

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int,
        compresslevel: int = 9,
        *,
        thread_minimum_size: int = 128 * 1024,  # 128 KiB
        exclude_content_types: tuple[str, ...] = DEFAULT_EXCLUDED_CONTENT_TYPES,
    ) -> None:
        super().__init__(app, minimum_size, exclude_content_types=exclude_content_types)

        self.compresslevel = compresslevel
        self.thread_minimum_size = thread_minimum_size
        self._compressor: zlib._Compress | None = None

    @property
    def compressor(self) -> zlib._Compress:
        if self._compressor is None:
            self._compressor = zlib.compressobj(self.compresslevel, zlib.DEFLATED, self.wbits)
        return self._compressor

    async def apply_compression(self, body: bytes, *, more_body: bool) -> bytes:
        if len(body) >= self.thread_minimum_size:
            # Compressing large chunks inline would block the event loop.
            limiter = _get_compression_capacity_limiter()
            return await anyio.to_thread.run_sync(self._compress_body, body, more_body, limiter=limiter)
        return self._compress_body(body, more_body)

    def _compress_body(self, body: bytes, more_body: bool) -> bytes:
        if more_body:
            return self.compressor.compress(body) + self.compressor.flush(zlib.Z_SYNC_FLUSH)
        return self.compressor.compress(body) + self.compressor.flush()


class GZipResponder(CompressedResponder):
    content_encoding = "gzip"
    wbits = 16 + zlib.MAX_WBITS


class DeflateResponder(CompressedResponder):
    content_encoding = "deflate"
    wbits = zlib.MAX_WBITS


async def unattached_send(message: Message) -> NoReturn:
    raise RuntimeError("send awaitable not set")  # pragma: no cover


def _normalize_content_types(content_types: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(content_type.partition(";")[0].strip().lower() for content_type in content_types)
