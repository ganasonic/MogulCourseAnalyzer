# OpenCV Zoo YOLOX model

The optional model is downloaded from the OpenCV Zoo `object_detection_yolox`
directory, revision `a74cdfad334b102cbd8daed769fbe1d4cb1c327a`.
That directory states that its files are licensed under Apache License 2.0.
The unmodified license is included in `LICENSE`. Model bytes are not in this Git
repository; the pinned URL, hash and size are in `models/yolox_manifest.json`.

Reference: https://github.com/opencv/opencv_zoo/tree/a74cdfad334b102cbd8daed769fbe1d4cb1c327a/models/object_detection_yolox

Our person-only adapter implements the documented YOLOX grid/stride output
decoding, top-left letterbox preprocessing and RGB input. It is a modified,
separate implementation for this application, not the upstream demonstration.

Tracking uses an independently implemented predictive two-confidence IoU
association algorithm inspired by ByteTrack. It is not the official ByteTrack
implementation and does not include its code, appearance embeddings or BoT-SORT.
Reference: https://github.com/ifzhang/ByteTrack
