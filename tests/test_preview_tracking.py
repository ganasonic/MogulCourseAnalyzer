import cv2
import numpy as np
from mogul.preview_tracking import PreviewTracking


def frames():
    rng=np.random.default_rng(42)
    image=np.zeros((120,160,3),np.uint8)
    image[30:65,40:65]=rng.integers(0,256,(35,25,3),dtype=np.uint8)
    moved=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,3]]),(160,120))
    record={'bbox':[40/160,30/120,65/160,65/120],'point':[52.5/160,65/120],
            'track_id':1,'run_id':'test','confidence':.8,'lane_id':'A'}
    return image,moved,record


def test_late_detections_are_carried_to_current_frame_by_observed_motion():
    first,second,record=frames(); tracker=PreviewTracking()
    tracker.frame(first,0,0); tracker.frame(second,100,1)
    result=tracker.correct([record],0,0)
    assert len(result)==1 and result[0]['interpolated']
    np.testing.assert_allclose(np.asarray(result[0]['bbox'])-record['bbox'],[4/160,3/120,4/160,3/120],atol=.002)
    assert record['bbox'][0]==40/160  # detector evidence is immutable


def test_failed_motion_or_expired_detection_hides_overlay():
    first,second,record=frames(); tracker=PreviewTracking(max_age_ms=150)
    tracker.frame(first,0,0); tracker.correct([record],0,0)
    assert tracker.frame(second,100,1)
    assert tracker.frame(second,200,2)==[]
    blank=np.zeros_like(first); tracker=PreviewTracking()
    tracker.frame(blank,0,0); tracker.correct([record],0,0)
    assert tracker.frame(blank,100,1)==[]


def test_seek_clears_motion_history_and_late_results_are_rejected():
    first,second,record=frames(); tracker=PreviewTracking()
    tracker.frame(first,5000,50); tracker.correct([record],5000,50)
    assert tracker.frame(second,0,0)==[]
    assert tracker.correct([record],5000,50)==[]
