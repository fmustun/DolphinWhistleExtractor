from __future__ import annotations

import struct


SHORT_LEN = 2
INT_LEN = 4
DOUBLE_LEN = 8
LONG_LEN = 8


class TonalHeader:
    """Minimal reader for NOAA silbido annotation headers.

    Adapted from the public `silbidopy` package distributed with the
    DCLDE 2011 NOAA archive.
    """

    def __init__(self, file_obj):
        self.HEADER_STR = "silbido!"
        self.TIME = 1
        self.FREQ = 1 << 1
        self.SNR = 1 << 2
        self.PHASE = 1 << 3
        self.SCORE = 1 << 4
        self.CONFIDENCE = 1 << 5
        self.RIDGE = 1 << 6
        self.TIMESTAMP = 1 << 7
        self.USERCOMMENT = 1 << 8
        self.SPECIES = 1 << 9
        self.CALL = 1 << 10
        self.DEFAULT = self.TIME | self.FREQ

        self.comment = None
        self.timestamp = None
        self.user_version = None
        self.bit_mask = None
        self.header_size = None
        self.version = None

        try:
            magic_str = str(file_obj.read(len(self.HEADER_STR)), "utf-8")
        except Exception:
            magic_str = ""

        if magic_str == self.HEADER_STR:
            self.version = int.from_bytes(file_obj.read(SHORT_LEN), byteorder="big")
            self.bit_mask = int.from_bytes(file_obj.read(SHORT_LEN), byteorder="big")
            self.user_version = int.from_bytes(file_obj.read(SHORT_LEN), byteorder="big")
            self.header_size = int.from_bytes(file_obj.read(INT_LEN), byteorder="big")

            header_used = 3 * SHORT_LEN + INT_LEN + len(self.HEADER_STR)
            remaining_len = self.header_size - header_used
            if remaining_len > 0:
                if (self.bit_mask & (self.USERCOMMENT | self.TIMESTAMP)) > 0:
                    if (self.bit_mask & self.USERCOMMENT) > 0:
                        comment_len = int.from_bytes(file_obj.read(2), byteorder="big")
                        self.comment = str(file_obj.read(comment_len), "utf-8")
                    else:
                        self.comment = ""

                    if (self.bit_mask & self.TIMESTAMP) > 0:
                        timestamp_len = int.from_bytes(file_obj.read(2), byteorder="big")
                        self.timestamp = str(file_obj.read(timestamp_len), "utf-8")
                else:
                    comment_len = int.from_bytes(file_obj.read(2), byteorder="big")
                    self.comment = str(file_obj.read(comment_len), "utf-8")
        else:
            self.bit_mask = self.DEFAULT
            self.user_version = -1
            self.version = -1

    def has_score(self) -> bool:
        return (self.bit_mask & self.SCORE) > 0

    def has_confidence(self) -> bool:
        return (self.bit_mask & self.CONFIDENCE) > 0

    def has_species(self) -> bool:
        return (self.bit_mask & self.SPECIES) > 0

    def has_call(self) -> bool:
        return (self.bit_mask & self.CALL) > 0

    def has_time(self) -> bool:
        return (self.bit_mask & self.TIME) > 0

    def has_freq(self) -> bool:
        return (self.bit_mask & self.FREQ) > 0

    def has_snr(self) -> bool:
        return (self.bit_mask & self.SNR) > 0

    def has_phase(self) -> bool:
        return (self.bit_mask & self.PHASE) > 0

    def has_ridge(self) -> bool:
        return (self.bit_mask & self.RIDGE) > 0


class tonalReader:
    """Reader for `.ann` files in the public DCLDE format."""

    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, "rb")
        self.hdr = TonalHeader(self.file)
        if self.hdr.user_version == -1:
            self.file.close()
            self.file = open(filename, "rb")

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.file.peek()) == 0:
            raise StopIteration

        confidence = 0.0
        score = 0.0
        call = ""
        species = ""
        graph_id = -1

        if self.hdr.has_confidence():
            confidence = struct.unpack(">d", self.file.read(DOUBLE_LEN))[0]
        if self.hdr.has_score():
            score = struct.unpack(">d", self.file.read(DOUBLE_LEN))[0]
        if self.hdr.has_species():
            str_len = int.from_bytes(self.file.read(2), byteorder="big")
            species = str(self.file.read(str_len), "utf-8")
        if self.hdr.has_call():
            str_len = int.from_bytes(self.file.read(2), byteorder="big")
            call = str(self.file.read(str_len), "utf-8")

        if self.hdr.version is not None and self.hdr.version > 2:
            graph_id = int.from_bytes(self.file.read(LONG_LEN), byteorder="big")

        point_count = int.from_bytes(self.file.read(INT_LEN), byteorder="big")
        tfnodes = []
        for _ in range(point_count):
            time = None
            freq = None
            snr = None
            phase = None
            ridge = False
            if self.hdr.has_time():
                time = struct.unpack(">d", self.file.read(DOUBLE_LEN))[0]
            if self.hdr.has_freq():
                freq = struct.unpack(">d", self.file.read(DOUBLE_LEN))[0]
            if self.hdr.has_snr():
                snr = struct.unpack(">d", self.file.read(DOUBLE_LEN))[0]
            if self.hdr.has_phase():
                phase = struct.unpack(">d", self.file.read(DOUBLE_LEN))[0]
            if self.hdr.has_ridge():
                ridge = struct.unpack(">d", self.file.read(DOUBLE_LEN))[0]
            tfnodes.append(
                {
                    "time": time,
                    "freq": freq,
                    "snr": snr,
                    "phase": phase,
                    "ridge": ridge,
                }
            )

        return {
            "species": species,
            "call": call,
            "graphId": graph_id,
            "confidence": confidence,
            "score": score,
            "tfnodes": tfnodes,
        }

    def getTimeFrequencyContours(self) -> list[list[tuple[float, float]]]:
        contours: list[list[tuple[float, float]]] = []
        for tonal in self:
            contour = []
            for node in tonal["tfnodes"]:
                time = node.get("time")
                freq = node.get("freq")
                if time is None or freq is None:
                    continue
                contour.append((float(time), float(freq)))
            contours.append(contour)
        return contours

