import sys
from pathlib import Path
import tempfile
import json
import datetime
from collections import OrderedDict
from typing import Optional, Union, Any, Generator, TextIO

from beartype import beartype
import numpy as np


DEBUG: int = 10
INFO: int = 20
WARN: int = 30
ERROR: int = 40

DISABLED: int = 50


class KVWriter(object):

    def writekvs(self, kvs):
        raise NotImplementedError(f"do whatever with {kvs}")


class SeqWriter(object):

    def writeseq(self, seq):
        raise NotImplementedError(f"do whatever with {seq}")


class HumanOutputFormat(KVWriter, SeqWriter):

    @beartype
    def __init__(self, path_or_io: Union[TextIO, Path]):
        if isinstance(path_or_io, Path):
            self.file = path_or_io.open("wt")
            self.own_file = True
        else:
            fmtstr = f"expected file or str, got {path_or_io}"
            assert hasattr(path_or_io, "read"), fmtstr
            self.file = path_or_io
            self.own_file = False

    @beartype
    def writekvs(self, kvs: dict[str, Union[int, float, np.ndarray]]):
        # create strings for printing
        key2str = {}
        for (key, val) in kvs.items():
            valstr = f"{val:<8.3g}" if isinstance(val, float) else str(val)
            key2str[self.truncate(key)] = self.truncate(valstr)

        # find max widths
        if len(key2str) == 0:
            # empty key-value dict; not sending warning nor stopping
            log("WARNING: tried to write empty key-value dict")
            return
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in key2str.items():
            key_padding = " " * (keywidth - len(key))
            val_padding = " " * (valwidth - len(val))
            lines.append(f"| {key}{key_padding} | {val}{val_padding} |")
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # flush the output to the file
        self.file.flush()

    @beartype
    @staticmethod
    def truncate(s: str) -> str:
        thres = 43
        return s[:40] + "..." if len(s) > thres else s

    @beartype
    def writeseq(self, seq: Generator[str, None, None]):
        for arg in seq:
            self.file.write(arg)
        self.file.write("\n")
        self.file.flush()

    @beartype
    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):

    @beartype
    def __init__(self, filename: Path):
        self.file = filename.open("wt")

    @beartype
    def writekvs(self, kvs: dict[str, Union[int, float, np.ndarray]]):
        for k, v in kvs.items():
            # if hasattr(v, "dtype"):
            if isinstance(v, np.ndarray):
                v_ = v.tolist()
                kvs[k] = float(v_)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    @beartype
    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):

    @beartype
    def __init__(self, filename: Path):
        self.file = filename.open("w+t")
        self.keys = []
        self.sep = ","

    @beartype
    def writekvs(self, kvs: dict[str, Union[int, float, np.ndarray]]):
        # add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


@beartype
def make_output_format(formatting: str, directory: Path, suffix: str = ""):
    directory.mkdir(parents=True, exist_ok=True)
    match formatting:  # python version >3.10 needed
        case "stdout":
            return HumanOutputFormat(sys.stdout)
        case "log":
            return HumanOutputFormat(directory / f"log{suffix}.txt")
        case "json":
            return JSONOutputFormat(directory / f"progress{suffix}.json")
        case "csv":
            return CSVOutputFormat(directory / f"progress{suffix}.csv")
        case _:
            raise ValueError(f"unknown formatting specified: {formatting}")


# frontend

def logkv(key, val):
    """Log a key-value pair with the current logger.
    This method should be called every iteration for the quantities to monitor.
    """
    if Logger.CURRENT is not None:
        Logger.CURRENT.logkv(key, val)


def logkvs(d):
    """Log a dictionary of key-value pairs with the current logger"""
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """Write all the key-values pairs accumulated in the logger
    to the write ouput format(s) (then flush the dictionary.
    """
    if Logger.CURRENT is not None:
        Logger.CURRENT.dumpkvs()


def getkvs():
    """Return the key-value pairs accumulated in the current logger"""
    if Logger.CURRENT is not None:
        return Logger.CURRENT.name2val
    return None


def log(*args, level=INFO):
    """Write the sequence of args, with no separators, to the console
    and output files (if an output file has been configured).
    """
    if Logger.CURRENT is not None:
        Logger.CURRENT.log(*args, level=level)


# create distinct functions fixed at all the values taken by `level`

def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """Set logging threshold on current logger"""
    if Logger.CURRENT is not None:
        Logger.CURRENT.set_level(level)


def get_dir():
    """Get directory to which log files are being written"""
    if Logger.CURRENT is not None:
        return Logger.CURRENT.get_dir()
    return None


# define aliases for higher-level language
record_tabular = logkv
dump_tabular = dumpkvs


# backend

class Logger(object):

    DEFAULT: Optional[Any] = None
    CURRENT: Optional[Any] = None

    @beartype
    def __init__(self, directory: Path,
                 output_formats: list[Union[HumanOutputFormat,
                                            JSONOutputFormat,
                                            CSVOutputFormat]]):
        self.name2val = OrderedDict()
        self.level: int = INFO
        self.directory: Path = directory
        self.output_formats: list[Union[HumanOutputFormat,
                                        JSONOutputFormat,
                                        CSVOutputFormat]] = output_formats

    @beartype
    def logkv(self, key: str, val: Union[int, float, np.ndarray]):
        self.name2val.update({key: val})

    @beartype
    def dumpkvs(self):
        if self.level == DISABLED:
            return
        for output_format in self.output_formats:
            if isinstance(output_format, KVWriter):
                output_format.writekvs(self.name2val)
        self.name2val.clear()

    @beartype
    def log(self, *args: Any, level: int = INFO):
        if self.level <= level:
            # if the current logger level is higher than
            # the `level` argument, don"t print to stdout
            self._log(args)

    @beartype
    def set_level(self, level: int):
        self.level = level

    @beartype
    def get_dir(self) -> Path:
        return self.directory

    @beartype
    def _log(self, args: tuple[Any, ...]):
        for output_format in self.output_formats:
            if isinstance(output_format, SeqWriter):
                x = (str(e) for e in args)
                output_format.writeseq(x)


@beartype
def configure(directory: Optional[Path] = None, format_strs: Optional[list[str]] = None):
    """Configure logger (called in configure_default_logger)"""
    if directory is None:
        directory = Path(tempfile.gettempdir())
        directory /= datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d-%H-%M-%S-%f_temp_log")
    else:
        assert isinstance(directory, Path)
        # make sure the provided directory exists
        Path(directory).mkdir(parents=True, exist_ok=True)
    if format_strs is None:
        format_strs = []
    # setup the output formats
    output_formats = [make_output_format(f, directory) for f in format_strs]
    Logger.CURRENT = Logger(directory=directory, output_formats=output_formats)


@beartype
def configure_default_logger():
    """Configure default logger"""
    # write to stdout by default
    format_strs = ["stdout"]
    # configure the current logger
    configure(format_strs=format_strs)  # makes Logger.CURRENT be not None anymore
    # logging successful configuration of default logger
    log("configuring default logger (logging to stdout only by default)")
    # define the default logger with the current logger
    Logger.DEFAULT = Logger.CURRENT


@beartype
def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT = Logger.DEFAULT
        log("resetting logger")
