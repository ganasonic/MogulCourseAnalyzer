import cv2
import numpy as np
from mogul.models import Calibration
from mogul.course_region import CourseRegion
from mogul.athlete_detector import AthleteDetector,Detection
from mogul.athlete_tracker import AthleteTracker
from mogul.preview_tracking import PreviewTracking


def course():
    c=Calibration(middle=[[.05,.05],[.95,.05],[.95,.95],[.05,.95]],middle_confirmed=True)
    c.configure_lanes(2)
    c.move_lane_corner(0,0,[.2,.15]); c.move_lane_corner(0,3,[.2,.85])
    c.move_lane_corner(1,1,[.8,.15]); c.move_lane_corner(1,2,[.8,.85])
    c.move_lane_corner(0,1,[.5,.15]); c.move_lane_corner(0,2,[.5,.85])
    return c


def detection(x,y):
    return Detection([x-.02,y-.08,x+.02,y],.9)


def test_roi_crops_and_masks_background_instead_of_using_the_whole_middle():
    c=course()
    c.move_lane_corner(0,0,[.3,.15]);c.move_lane_corner(1,1,[.7,.15])
    region=CourseRegion(c)
    image=np.full((200,320,3),255,np.uint8)
    crop,(x,y)=region.crop(image)
    assert crop.shape[0]*crop.shape[1]<200*320*.8
    assert x>0 and y>0
    # A rectangular crop's corners are outside the trapezoidal lane union.
    assert np.any(crop==114) and np.any(crop==255)
    assert not region.contains([.1,.5])  # inside middle, outside all lanes
    assert not region.can_start([.82,.4])


def test_staff_cannot_create_tracks_but_existing_skier_can_cross_boundary():
    region=CourseRegion(course()); tracker=AthleteTracker(min_hits=1)
    first=tracker.update([detection(.79,.4),detection(.82,.7)],0,can_start=region.can_start)
    assert len(first)==1 and first[0].id==1
    second=tracker.update([detection(.82,.43),detection(.82,.7)],100,can_start=region.can_start)
    assert len(second)==1 and second[0].id==1 and second[0].foot[0]>.8
    assert tracker.next_id==2
    assert region.can_start([.35,.13])  # narrow upstream entry band
    assert not region.can_start([.35,.9])  # no new downstream staff tracks


def test_detector_crop_coordinates_map_back_to_video_coordinates():
    image=np.zeros((200,320,3),np.uint8); region=CourseRegion(course())
    crop,(x,y)=region.crop(image); h,w=crop.shape[:2]
    expected=np.array([.3,.3,.35,.4])
    local=(expected*[320,200,320,200]-[x,y,x,y])/[w,h,w,h]
    detector=object.__new__(AthleteDetector)
    observed=[]
    def detect(img,middle,tiled):
        observed.append(img.shape)
        return [Detection(local.tolist(),.8)]
    detector.detect=detect
    result=detector.detect_course(image,region,tiled=True)
    np.testing.assert_allclose(result[0].bbox,expected)
    assert observed==[crop.shape]


def test_optical_flow_never_carries_staff_outside_course():
    image=np.random.default_rng(7).integers(0,255,(120,160,3),dtype=np.uint8)
    region=CourseRegion(course()); tracker=PreviewTracking(region=region)
    tracker.frame(image,0,0)
    records=[dict(bbox=detection(x,.4).bbox,track_id=i,run_id='test',confidence=.9)
             for i,x in enumerate([.35,.82],1)]
    result=tracker.correct(records,0,0)
    assert len(result)==1 and result[0]['track_id']==1
    tracker.advance=lambda *args:[dict(records[0],bbox=detection(.82,.4).bbox)]
    assert tracker.frame(image,100,1)==[]


def test_cropped_optical_flow_keeps_video_coordinates():
    image=np.zeros((120,160,3),np.uint8)
    image[30:65,40:65]=np.random.default_rng(42).integers(0,255,(35,25,3),dtype=np.uint8)
    moved=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,3]]),(160,120))
    tracker=PreviewTracking(region=CourseRegion(course()))
    record={'bbox':[40/160,30/120,65/160,65/120],'track_id':1}
    tracker.frame(image,0,0);tracker.frame(moved,100,1)
    assert tracker.history[-1][2].size<120*160
    result=tracker.correct([record],0,0)
    assert len(result)==1
    np.testing.assert_allclose(np.asarray(result[0]['bbox'])-record['bbox'],[4/160,3/120,4/160,3/120],atol=.002)
