import sys
from pathlib import Path
import hashlib
import time

import numpy as np
import cv2
from beartype import beartype

from helpers import logger


@beartype
def record_video(save_dir: Path, name: str, obs: np.ndarray):
    """Record a video from samples collected at evalutation time."""
    # unstack the frames if stacked, while leaving colors unaltered
    frames = np.split(obs, 1, axis=-1)
    frames = np.concatenate(np.array(frames), axis=0)
    frames = [np.squeeze(a, axis=0)
              for a in np.split(frames, frames.shape[0], axis=0)]

    # create OpenCV video writer
    vname = f"render-{name}"
    frame_size = (obs.shape[-2],
                 obs.shape[-3])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        filename=f"{save_dir / vname}.mp4",
        fourcc=fourcc,
        fps=25,
        frameSize=frame_size,
        isColor=True,
    )

    for frame in frames:
        # add frame to video
        writer.write(frame)
    writer.release()
    cv2.destroyAllWindows()
    # delete the object
    del frames

    logger.info(f"video::{vname}::dumped")


class OpenCVImageViewer(object):
    """Viewer used to render simulations."""

    @beartype
    def __init__(self, *, q_to_exit=True):
        self._q_to_exit = q_to_exit
        # create unique identifier
        hash_ = hashlib.sha1()
        hash_.update(str(time.time()).encode("utf-8"))
        # create window
        self._window_name = str(hash_.hexdigest()[:20])
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._isopen = True

    @beartype
    def __del__(self):
        cv2.destroyWindow(self._window_name)
        self._isopen = False

    @beartype
    def imshow(self, img):
        # convert image to BGR format
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        # listen for escape key, then exit if pressed
        if cv2.waitKey(1) == ord("q") and self._q_to_exit:
            sys.exit()

    @beartype
    @property
    def isopen(self):
        return self._isopen

    @beartype
    def close(self):
        pass
