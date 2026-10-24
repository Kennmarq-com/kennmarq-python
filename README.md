# kennmarq

Mark AI pictures, songs and clips for the EU AI Act and California's AI Transparency Act in one call. Each gets an invisible watermark
(TrustMark for pictures, AudioSeal for sound, Video Seal for frames), and signed, time-stamped C2PA
Content Credentials that say it was made with AI. Kennmarq.com helps you
meet the Act's marking duty (Art. 50(2)) and the Code of Practice's two-layer measure; you stay
responsible for your system.

```bash
pip install kennmarq                 # the cloud API: no dependencies
pip install "kennmarq[local]"        # local mode: files never leave your machine
pip install "kennmarq[safety]"       # the safety check on your machine too
```

A free key comes with 1,000 marks a month and every feature: [kennmarq.com](https://kennmarq.com).
The SDK reads it from `KENNMARQ_API_KEY`.

## Cloud

```python
from kennmarq import Kennmarq

k = Kennmarq()
result = k.mark("render.png", "render.marked.png", source="created", model="FLUX.1 schnell")
print(result.mark_id, result.watermark_id, result.marks_used)
```

The picture goes to Kennmarq.com's servers in Zurich, which mark it in memory and keep nothing.

Add `label=True` to draw the EU's visible AI label ("AI GENERATED", or "AI MODIFIED" for edits) into a
picture or every frame of a clip, before the watermark; `deepfake=True` turns it on by itself, and
`label_language`, `label_position`, `label_size` and `label_style` shape it
([docs](https://kennmarq.com/docs/api/mark#labels)). Labels are drawn in the cloud.

## Songs and clips

```python
song = k.mark("track.mp3", "track.marked.mp3", model="Song Model 2")
print(song.marks_counted, song.audio_watermark)   # five marks per started minute; "claimed"

clip = k.mark("clip.mp4", "clip.marked.mp4", check="ai-act")   # waits for the job, then downloads
print(clip.watermark_id, clip.marks_counted)      # five marks per started second
```

Songs (MP3, WAV, M4A, up to 15 minutes) carry Meta's AudioSeal in the sound, shaped so no one hears
it; clips (MP4 at up to 1080p, 4,500 frames: 2.5 minutes at 30 frames a second) carry Video Seal in every frame and AudioSeal in their
sound. Both are marked in the cloud. A clip is a job: `mark()` waits for it and downloads the marked
clip; `mark(..., wait=False)` returns at once, and `wait_for(mark_id, dest)` finishes later.

## Local mode

```python
result = k.mark_file("render.png", "render.marked.png", source="created", model="FLUX.1 schnell")
```

The watermark and the Content Credentials are made on your machine. Only the manifest goes to
Kennmarq.com: about 13 KB of claim and assertions, never pixels. Kennmarq.com signs and time-stamps it, and
the SDK puts it back in your file. Before writing anything, the SDK checks the file reads as signed by
Kennmarq.com, with a validated time-stamp and the watermark in place; if not, it raises `InvalidSignedFile`
and writes nothing. Each file counts one mark.

- Credentials your file already carries (from an upstream model's API, say) are kept as an
  ingredient, as the Code of Practice asks. They travel with the manifest, with any thumbnail their
  signer put in them; `keep_input_credentials=False` keeps and sends none.
- The watermark's model (TrustMark Q, 65 MB) downloads once, to `~/.cache/kennmarq/models` or
  `KENNMARQ_MODELS_DIR`, and is checked against its pinned SHA-256.
- Where c2pa-python has no build for your system (Windows, for now), `mark_file` makes the watermark
  on your machine and sends the watermarked picture to the cloud for its manifest (`mode == "hybrid"`).
- Without a key, `watermark_file()` adds the watermark and the IPTC AI label alone, with no
  Content Credentials.

## The safety check

```python
answer = k.check("render.png", policy="ai-act")              # cloud: two marks
answer = k.check("render.png", policy="ai-act", local=True)  # here: nothing sent, nothing counted
if answer["verdict"] == "block":
    ...
```

It screens a picture for what the AI Act's new bans are about (Art. 5(1)(ba) and (bb)): sexual content
with someone who appears to be a child, and with a realistic face, which needs the person's consent.
`ai-act` blocks the first and flags the second; `safe-only` also blocks anything explicit. It returns
a verdict and a reason, never an age or an identity. It's a risk signal with published error rates
([docs](https://kennmarq.com/docs/api/check)), not a guarantee: you decide, and stay responsible.
Locally its models download once (about 2.3 GB). `mark()` and `mark_file()` take `check=` too: a block
raises `ContentBlocked` and nothing is marked.

## Check a file

```python
k.verify("render.marked.png")    # free, no key needed
```

What its Content Credentials and its watermark say, and the credential the watermark leads to when the
metadata was stripped.

## Errors

API errors raise `KennmarqError` with the API's `code`, `status` and `request_id`
([docs](https://kennmarq.com/docs/errors)). `ContentBlocked` carries the check's answer in `safety`.

MIT-licensed. The paid part is Kennmarq.com's signature, the time-stamps and the stored credentials.
The face model and the prompt embeddings bundled for the safety check carry their own licences:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
