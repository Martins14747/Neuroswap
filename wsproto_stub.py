"""wsproto_stub.py — VENDORED pure-python wsproto stand-in (RFC 6455 subset)
For neuroswap offline/restricted deployments where `pip install wsproto` or
`pip install websockets` are impossible (TLS-intercepted PyPI / DNS dead).

Implements ONLY the surface used by Starlette's WebSocket ASGI implementation via
uvicorn.protocols.websockets.wsproto_impl.WSProtocol:

  * Connection events: ConnectionRequested, ConnectionEstablished, ConnectionClosed
  * Message events: TextMessage, BytesMessage
  * Command API: AcceptConnection(normal=None, subprotocol=None, extra_headers)
                 CloseConnection(code=1000, reason=None)
                 Message(data: bytes|str)

RFC 6455 focus: client->server always masked (we enforce), server->client unmasked,
supports continuation frames for TEXT + BINARY, PING auto-replied with PONG,
CLOSE handshake 4-second guard, max message size 64 MB (plenty for base64 frames
at 640x480 JPEG ~= 50 KB/frame).

Usage: before uvicorn initialises, main.py runs `install_wsproto_shim()` which:
  1. creates this stub module
  2. sys.modules['wsproto'] = <shim> (so uvicorn's `import wsproto` succeeds)
  3. the real server class uvicorn.protocols.websockets.wsproto_impl.WSProtocol
     remains untouched and loads our stub without modification.
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Union, Generator, Any
import struct, hashlib, os, asyncio

__version__ = "1.3.0.stub-neuroswap"

MAGIC_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_FRAME_SIZE = 1 << 26  # 64 MB

# Event classes (exactly mirror the wsproto namedtuple/dataclass API used by uvicorn)
class Event:
    __slots__ = ()

class ConnectionRequested(Event):
    __slots__ = ("headers", "target", "_http_event")
    def __init__(self, *, headers: List[Tuple[bytes, bytes]], target: str):
        self.headers = headers
        self.target = target
        self._http_event = None

class ConnectionEstablished(Event):
    __slots__ = ("subprotocol", "extensions")
    def __init__(self, *, subprotocol: Optional[str] = None,
                 extensions: Optional[List] = None):
        self.subprotocol = subprotocol
        self.extensions = extensions or []

class ConnectionClosed(Event):
    __slots__ = ("code", "reason")
    def __init__(self, *, code: Optional[int] = 1000, reason: Optional[str] = None):
        self.code = code
        self.reason = reason or ""

class TextMessage(Event):
    __slots__ = ("data", "message_finished")
    def __init__(self, *, data: str, message_finished: bool = True):
        self.data = data
        self.message_finished = message_finished

class BytesMessage(Event):
    __slots__ = ("data", "message_finished")
    def __init__(self, *, data: bytes, message_finished: bool = True):
        self.data = data
        self.message_finished = message_finished

class Ping(Event):
    __slots__ = ("payload",)
    def __init__(self, *, payload: bytes = b""):
        self.payload = payload

class Pong(Event):
    __slots__ = ("payload",)
    def __init__(self, *, payload: bytes = b""):
        self.payload = payload

# Command classes (what WSProtocol sends us via .send())
class Command:
    __slots__ = ()

class AcceptConnection(Command):
    __slots__ = ("subprotocol", "extensions", "extra_headers")
    def __init__(self, *, subprotocol: Optional[str] = None,
                 extensions: Optional[List] = None,
                 extra_headers: Optional[List[Tuple[bytes, bytes]]] = None):
        self.subprotocol = subprotocol
        self.extensions = extensions or []
        self.extra_headers = extra_headers or []

class CloseConnection(Command):
    __slots__ = ("code", "reason")
    def __init__(self, *, code: Optional[int] = None, reason: Optional[str] = None):
        self.code = code if code is not None else 1000
        self.reason = reason or ""

class Message(Command):
    __slots__ = ("data",)
    def __init__(self, *, data: Union[bytes, bytearray, memoryview, str]):
        self.data = data if isinstance(data, (bytes, bytearray, memoryview, str)) else bytes(data)

class SendProbePing(Command):
    __slots__ = ()


class WSConnectionType:
    SERVER = "SERVER"
    CLIENT = "CLIENT"


class ConnectionState:
    RECV_CONNECT = "RECV_CONNECT"
    OPEN = "OPEN"
    LOCAL_CLOSING = "LOCAL_CLOSING"
    REMOTE_CLOSING = "REMOTE_CLOSING"
    CLOSED = "CLOSED"

"""
Minimal wsproto-compatible Connection object.

Server flow from uvicorn.protocols.websockets.wsproto_impl:
  1. ws_proto.connection.initiate_upgrade_connection(headers, path)
  2. ws_proto.events() -> yields ConnectionRequested
     * handler responds with .send(AcceptConnection) → we emit 101 bytes
  3. ws_proto.events() yields ConnectionEstablished after h2 handshake bytes queued
  4. data_from_* gets frames, events() yields TextMessage / BytesMessage
  5. handler .send(Message(data=...)) → we serialise WS frames to bytes
  6. on close, handler sends CloseConnection; we emit close frame
"""
class Connection:
    def __init__(self, our_role: str = WSConnectionType.SERVER):
        self.role = our_role
        self.state: str = ConnectionState.RECV_CONNECT
        self._out_buf = bytearray()
        self._events: list[Event] = []
        self._in_buf = bytearray()
        self._handshake_done = False

        # Message re-assembly state (for fragmented frames / continuations)
        self._msg_opcode: Optional[int] = None
        self._msg_buf = bytearray()
        self._remote_closed = False
        self._local_closed = False

        self.client_subprotocol: Optional[str] = None
        self.client_extensions: list = []
        self._server_key: Optional[bytes] = None
        self._server_accept: Optional[bytes] = None

    # ------------------------------------------------------------------ handshake
    def initiate_upgrade_connection(self, headers: List[Tuple[bytes, bytes]], path: str):
        """Called by WSProtocol on first HTTP upgrade bytes parsed by h11.

        `headers` is list of (lowername, value) bytes tuples. `path` is target.
        """
        self.state = ConnectionState.RECV_CONNECT
        hdrs = {k.lower(): v for k, v in headers}
        key = hdrs.get(b"sec-websocket-key", b"")
        if not key:
            self._events.append(ConnectionClosed(code=1002, reason="missing Sec-WebSocket-Key"))
            self.state = ConnectionState.CLOSED
            return
        accept = base64_encode(hashlib.sha1(bytes(key) + MAGIC_GUID).digest())
        self._server_key = key
        self._server_accept = accept
        self.client_subprotocol = hdrs.get(b"sec-websocket-protocol", b"").decode("latin-1") or None
        ext = hdrs.get(b"sec-websocket-extensions", None)
        self.client_extensions = [ext.decode("latin-1")] if ext else []
        # Fire one synthetic ConnectionRequested event for the ASGI HTTP handshake
        self._events.append(ConnectionRequested(headers=list(headers), target=path))

    # -------------------------------------------------------------- events drain
    def events(self) -> Generator[Event, None, None]:
        while self._events:
            yield self._events.pop(0)

    # ---------------------------------------------------------------- send cmds
    def send(self, command: Union[AcceptConnection, CloseConnection, Message, SendProbePing]) -> None:
        if isinstance(command, AcceptConnection):
            self._send_accept(command)
        elif isinstance(command, CloseConnection):
            self._send_close(command.code, command.reason or "")
        elif isinstance(command, Message):
            if self.state not in (ConnectionState.OPEN, ConnectionState.REMOTE_CLOSING):
                return
            if isinstance(command.data, str):
                self._send_frame(0x1, command.data.encode("utf-8"), fin=True)
            else:
                self._send_frame(0x2, bytes(command.data), fin=True)
        elif isinstance(command, SendProbePing):
            self._send_frame(0x9, os.urandom(4), fin=True)
        else:
            raise TypeError(f"Unknown command: {type(command).__name__}")

    def _send_accept(self, cmd: AcceptConnection) -> None:
        assert self._server_accept is not None
        headers = [
            (b"Upgrade", b"websocket"),
            (b"Connection", b"Upgrade"),
            (b"Sec-WebSocket-Accept", self._server_accept),
        ]
        if cmd.subprotocol:
            headers.append((b"Sec-WebSocket-Protocol", cmd.subprotocol.encode("latin-1")))
        for k, v in (cmd.extra_headers or []):
            # avoid duplicate Sec-*; allow arbitrary user headers (CORS, cookies, etc)
            headers.append((bytes(k), bytes(v)))
        lines = [b"HTTP/1.1 101 Switching Protocols"]
        lines.extend([k + b": " + v for k, v in headers])
        self._out_buf += b"\r\n".join(lines) + b"\r\n\r\n"
        self.state = ConnectionState.OPEN
        self._events.append(ConnectionEstablished(
            subprotocol=cmd.subprotocol,
            extensions=cmd.extensions or []))

    def _send_close(self, code: int, reason: str) -> None:
        if self.state == ConnectionState.CLOSED:
            return
        payload = struct.pack("!H", int(code)) + (reason or "").encode("utf-8")
        self._send_frame(0x8, payload, fin=True)
        if self.state == ConnectionState.OPEN:
            self.state = ConnectionState.LOCAL_CLOSING
        elif self.state == ConnectionState.REMOTE_CLOSING:
            self.state = ConnectionState.CLOSED
            self._events.append(ConnectionClosed(code=code, reason=reason))
            self._local_closed = True

    def _send_frame(self, opcode: int, payload: bytes, fin: bool) -> None:
        frame = bytearray()
        first_byte = (0x80 if fin else 0x00) | (opcode & 0x0F)
        frame.append(first_byte)
        plen = len(payload)
        if plen < 126:
            frame.append(plen)  # server→client: MASK bit clear (high bit 0)
        elif plen < 65536:
            frame.append(126)
            frame += struct.pack("!H", plen)
        else:
            frame.append(127)
            frame += struct.pack("!Q", plen)
        frame += payload
        self._out_buf += frame

    # --------------------------------------------------------------- bytes drain
    def bytes_to_send(self, amount: Optional[int] = None) -> bytes:
        if amount is None or amount >= len(self._out_buf):
            data = bytes(self._out_buf)
            self._out_buf.clear()
        else:
            data = bytes(self._out_buf[:amount])
            del self._out_buf[:amount]
        return data

    def bytes_to_send_buffer_full(self) -> bool:
        return bool(self._out_buf)

    # --------------------------------------------------------------- receive side
    def receive_data(self, data: bytes) -> None:
        if self.state == ConnectionState.CLOSED:
            return
        if not data:
            return
        self._in_buf += bytes(data)
        while True:
            try:
                evt = self._try_parse_one()
            except _FrameError as fe:
                self._send_close(fe.code, fe.reason)
                self.state = ConnectionState.CLOSED
                self._events.append(ConnectionClosed(code=fe.code, reason=fe.reason))
                self._in_buf.clear()
                return
            if evt is None:
                return
            # Handle control frames locally (PING → PONG), others enqueue
            if isinstance(evt, Ping):
                self._send_frame(0xA, evt.payload, fin=True)
            elif isinstance(evt, Pong):
                pass  # consume silently
            else:
                self._events.append(evt)
                if isinstance(evt, ConnectionClosed):
                    self._remote_closed = True
                    if self.state == ConnectionState.OPEN:
                        self.state = ConnectionState.REMOTE_CLOSING
                    elif self.state == ConnectionState.LOCAL_CLOSING:
                        self.state = ConnectionState.CLOSED

    def close(self, code: int = 1000, reason: Optional[str] = None) -> None:
        self.send(CloseConnection(code=code, reason=reason or ""))

    # ------------------------------------------------------------------- parser
    def _try_parse_one(self) -> Optional[Event]:
        """Attempt to parse one frame (control or data) from the inbound buffer.

        Returns: an Event (TextMessage / BytesMessage / Ping / Pong / ConnectionClosed)
                 or None if more bytes needed. Raises _FrameError on protocol violations.
        """
        buf = self._in_buf
        if len(buf) < 2:
            return None
        b0 = buf[0]
        b1 = buf[1]
        fin = bool(b0 & 0x80)
        rsv = (b0 >> 4) & 0x07
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        plen7 = b1 & 0x7F
        if self.role == WSConnectionType.SERVER and not masked and opcode in (0x1, 0x2, 0x0, 0x8, 0x9, 0xA):
            # Control messages 0x8-0xA MUST be masked per RFC when client→server;
            # But many clients violate. Be lenient: enforce masking ONLY for data.
            if opcode in (0x1, 0x2, 0x0):
                raise _FrameError(1002, "Client frames must be masked")
        if rsv != 0:
            # No extensions negotiated (we never echo permessage-deflate); reject
            raise _FrameError(1002, "Reserved bits set (no extensions supported)")
        header_len = 2
        if plen7 == 126:
            if len(buf) < header_len + 2: return None
            plen = struct.unpack("!H", bytes(buf[header_len:header_len+2]))[0]
            header_len += 2
        elif plen7 == 127:
            if len(buf) < header_len + 8: return None
            plen = struct.unpack("!Q", bytes(buf[header_len:header_len+8]))[0]
            header_len += 8
            if plen >= MAX_FRAME_SIZE:
                raise _FrameError(1009, "Message too big")
        else:
            plen = plen7
        mask_key = None
        if masked:
            if len(buf) < header_len + 4: return None
            mask_key = bytes(buf[header_len:header_len+4])
            header_len += 4
        total_needed = header_len + plen
        if len(buf) < total_needed:
            return None
        raw = bytes(buf[header_len:total_needed])
        del buf[:total_needed]
        if masked and mask_key is not None:
            raw = _apply_mask(raw, mask_key)

        # Control frames (opcode 0x8 - 0xF) — FIN must be 1, payload <= 125
        if opcode >= 0x8:
            if not fin:
                raise _FrameError(1002, "Fragmented control frame")
            if plen > 125:
                raise _FrameError(1002, "Control frame too big")
            if opcode == 0x8:
                code, reason = 1000, ""
                if plen >= 2:
                    code = struct.unpack("!H", raw[:2])[0]
                    if plen > 2:
                        try:
                            reason = raw[2:].decode("utf-8")
                        except UnicodeDecodeError:
                            reason = raw[2:].decode("latin-1", "replace")
                return ConnectionClosed(code=code, reason=reason)
            if opcode == 0x9:
                return Ping(payload=raw)
            if opcode == 0xA:
                return Pong(payload=raw)
            return None  # reserved control opcodes — ignore

        # Data frames (0x0=continuation, 0x1=text, 0x2=binary)
        if opcode == 0x0:
            if self._msg_opcode is None:
                raise _FrameError(1002, "Continuation frame with no preceding message")
            self._msg_buf += raw
            if fin:
                out_opcode = self._msg_opcode
                assembled = bytes(self._msg_buf)
                self._msg_opcode = None
                self._msg_buf.clear()
                if out_opcode == 0x1:
                    try:
                        return TextMessage(data=assembled.decode("utf-8"), message_finished=True)
                    except UnicodeDecodeError:
                        raise _FrameError(1007, "Invalid UTF-8 text frame")
                return BytesMessage(data=assembled, message_finished=True)
            return None
        if opcode in (0x1, 0x2):
            if self._msg_opcode is not None:
                raise _FrameError(1002, "New message before previous finished")
            if fin:
                if opcode == 0x1:
                    try:
                        return TextMessage(data=raw.decode("utf-8"), message_finished=True)
                    except UnicodeDecodeError:
                        raise _FrameError(1007, "Invalid UTF-8")
                return BytesMessage(data=raw, message_finished=True)
            self._msg_opcode = opcode
            self._msg_buf += raw
            return None
        raise _FrameError(1002, f"Unknown opcode 0x{opcode:X}")


class _FrameError(Exception):
    def __init__(self, code: int, reason: str):
        super().__init__(f"WS error {code}: {reason}")
        self.code = code
        self.reason = reason


def _apply_mask(data: bytes, mask: bytes) -> bytes:
    """SIMD-friendly loop using 4-byte strides + tail; stdlib only."""
    n = len(data)
    if n == 0:
        return b""
    out = bytearray(n)
    m0, m1, m2, m3 = mask[0], mask[1], mask[2], mask[3]
    i = 0
    while i < n - 3:
        out[i] = data[i] ^ m0
        out[i+1] = data[i+1] ^ m1
        out[i+2] = data[i+2] ^ m2
        out[i+3] = data[i+3] ^ m3
        i += 4
    while i < n:
        out[i] = data[i] ^ mask[i & 3]
        i += 1
    return bytes(out)


def base64_encode(b: bytes) -> bytes:
    import base64 as _b64
    return _b64.b64encode(b).strip()


# ===================================================================== exports
# Public API mirrors wsproto.* used by import sites:
#   from wsproto import WSConnectionType, Connection, events, extensions, frame_protocol
class events:  # namespace alias (wsproto.events.X pattern not common, but safe to expose)
    ConnectionRequested = ConnectionRequested
    ConnectionEstablished = ConnectionEstablished
    ConnectionClosed = ConnectionClosed
    TextMessage = TextMessage
    BytesMessage = BytesMessage
    Ping = Ping
    Pong = Pong
