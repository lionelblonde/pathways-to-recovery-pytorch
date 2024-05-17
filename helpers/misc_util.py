import time
from contextlib import contextmanager

from beartype import beartype
from termcolor import colored

from helpers import logger


@beartype
def prettify_numb(n: int) -> str:
    """Display an integer number of millions, ks, etc."""
    m, k = divmod(n, 1_000_000)
    k, u = divmod(k, 1_000)
    return colored(f"{m}M {k}K {u}U", "red", attrs=["reverse"])


@beartype
def prettify_time(seconds: int) -> str:
    """Print the number of seconds in human-readable format.
    e.g. "2 days", "2 hours and 37 minutes", "less than a minute".
    """
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    days = hours // 24
    hours %= 24

    def helper(count, name):
        trailing_s = "s" if count > 1 else ""
        return f"{count} {name}{trailing_s}"

    # display only the two greater units (days and hours, hours and minutes, minutes and seconds)
    if days > 0:
        message = helper(days, "day")
        if hours > 0:
            message += " and " + helper(hours, "hour")
        return message
    if hours > 0:
        message = helper(hours, "hour")
        if minutes > 0:
            message += " and " + helper(minutes, "minute")
        return message
    if minutes > 0:
        return helper(minutes, "minute")

    # finally, if none of the previous conditions is valid
    return "less than a minute"


@beartype
@contextmanager
def timed_ctx(text: str):
    pre_mess = f"::{text}::"
    logger.info(colored(pre_mess, "magenta", attrs=["underline", "bold"]))
    tstart = time.time()
    yield
    tot_time = time.time() - tstart
    post_mess = f"done in {tot_time}secs".rjust(50, ":")
    logger.info(colored(post_mess, "magenta"))


@beartype
def log_iter_info(cur_iter: int, tot_num_iters: int, tstart: float):
    """Display the current iteration and elapsed time"""
    elapsed = prettify_time(int(time.time() - tstart))
    mess = f"iter [{cur_iter}/{tot_num_iters}] <- elapsed time: {elapsed}".rjust(75, ":")
    logger.info(colored(mess, "magenta"))
