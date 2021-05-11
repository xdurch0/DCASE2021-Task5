def time_to_frame(time: float, fps: float) -> int:
    """Convert time in seconds to frame number.

    Parameters:
        time: The time in seconds.
        fps: Number of frames per second.

    Returns:
        Frame index corresponding to time. Rounded down!!

    """
    return int(time * fps)


EVENT_ESTIMATES = {"AMRE": 10,
                   "BBWA": 6,
                   "BTBW": 10,
                   "COYE": 8,
                   "OVEN": 6,
                   "RBGR": 0,
                   "SWTH": 0,
                   "GCTH": 0,
                   "CHSP": 5,
                   "SAVS": 5,
                   "WTSP": 0}


def correct_events(start, end, cls):
    correction = EVENT_ESTIMATES[cls]
    return start + correction, end - correction

