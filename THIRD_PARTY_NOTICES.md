# Third-party notices

The safety check (`kennmarq/local/safety/`) ships two small files made by others. Its larger models
aren't part of this package: `fetch()` downloads them from their publishers, each pinned to a commit and
checked against its SHA-256, under their own licences (listed in `kennmarq/local/safety/models.py`).

## YuNet face detector

`kennmarq/local/safety/assets/face_detection_yunet_2023mar.onnx`, from OpenCV Zoo
(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), unchanged.

```
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Prompt embeddings

`kennmarq/local/safety/assets/age_gate_prompts.npz` holds the embeddings of `age.py`'s photo and
drawing prompts, computed with the text half of google/siglip-base-patch16-224
(https://huggingface.co/google/siglip-base-patch16-224), licensed under the Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0).
