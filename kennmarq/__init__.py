"""Kennmarq.com's Python SDK: mark AI pictures, songs and clips for the EU AI Act with an invisible
watermark and signed, time-stamped C2PA Content Credentials, in the cloud or (pictures) on your own machine.

    from kennmarq import Kennmarq
    k = Kennmarq()                                    # reads KENNMARQ_API_KEY
    k.mark("render.png", "render.marked.png")         # cloud
    k.mark("song.mp3", "song.marked.mp3")             # a song, in the cloud
    k.mark("clip.mp4", "clip.marked.mp4")             # a clip: a job, waited for and downloaded
    k.mark_file("render.png", "render.marked.png")    # local: only the manifest leaves this machine
"""
from ._version import __version__
from .client import Kennmarq, MarkResult
from .errors import ContentBlocked, InvalidSignedFile, KennmarqError, LocalModeUnavailable, MissingApiKey

__all__ = ["ContentBlocked", "InvalidSignedFile", "Kennmarq", "KennmarqError", "LocalModeUnavailable", "MarkResult",
           "MissingApiKey", "__version__"]
