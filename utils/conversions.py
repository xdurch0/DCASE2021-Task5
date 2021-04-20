def time_to_frame(time: float, fps: float) -> int:
    """Convert time in seconds to frame number.

    Parameters:
        time: The time in seconds.
        fps: Number of frames per second.

    Returns:
        Frame index corresponding to time. Rounded down!!

    """
    return int(time * fps)
