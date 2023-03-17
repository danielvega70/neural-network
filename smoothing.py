import numpy as np

def smooth(x, window_len=11, window='hanning'):
    """
    Smooth the input signal using a window of the requested size and type.

    Parameters
    ----------
    x : array_like
        The input signal.
    window_len : int, optional
        The length of the smoothing window. Must be odd. Default is 11.
    window : str, optional
        The type of window to use. Must be one of 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'. Default is 'hanning'.

    Returns
    -------
    y : ndarray
        The smoothed signal.

    Notes
    -----
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) at both ends so that boundary effect are minimized
    in the beginning and end part of the signal.

    Examples
    --------
    >>> t = np.linspace(-2, 2, 0.1)
    >>> x = np.sin(t) + np.random.randn(len(t)) * 0.1
    >>> y = smooth(x)

    See also
    --------
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    References
    ----------
    .. [1] https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    if x.ndim != 1:
        raise ValueError("Input signal must be 1-dimensional")
    if x.size < window_len:
        raise ValueError("Input signal size must be at least window size")
    if window_len < 3:
        return x

    # Create the window
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)

    # Convolve the signal with the window
    y = np.convolve(w / w.sum(), x, mode='same')

    # Trim the reflected copies at the beginning and end
    half_window = window_len // 2
    y = y[half_window:-half_window]

    return y
