from dataclasses import dataclass, field
from datetime import datetime
import glob
import math
import random
import sys
sys.path.append("/age_gender/detection/camera_action")
print(sys.path)
from typing import List, Dict, Tuple, Optional, Set
import cv2
import numpy as np
from collections import defaultdict, deque
from python_app.module.mapper_bytetrack import HandPersonMapper, get_expand_timestamp, get_timestamp
from python_app.module.alcohol_tensorrt import AlcoholBottleTracker
from python_app.module.hand_manager import HandManager, HandTrack
from collections import Counter

from python_app.module.bin_file_reader_skip import BinFileReader
import pytz
hanoi_tz = pytz.timezone("Asia/Ho_Chi_Minh")

@dataclass
class TouchEvent:
    roi_idx : int
    person_id : int
    frame_idx : int
    time_out : int
    id : str = None

    def __post_init__(self):
        self.id = f"{self.roi_idx}_{self.frame_idx}"

config = {}

from typing import Dict, List


@dataclass
class WashingAction:
    frame_id: int
    one_hand: list = field(default_factory=list)
    overlap: list = field(default_factory=list)


    def get_total(self):
        return len(self.one_hand) + len(self.overlap)
    
    def get_one_hand(self):
        return len(self.one_hand)

    def get_overlap(self):
        return len(self.overlap)


@dataclass
class ShakingAction:
    distance: float
    max_distance: float
    frame_id: int
    
@dataclass
class ShakingResult:
    data: List[ShakingAction] = field(default_factory=list)

class ShakingDetector:
    def __init__(self,
                 d_min: float = 20.0,
                 d_max: float = 400.0,
                 s_min: float = 170.0,
                 ratio_min: float = 1.85,
                 min_duration: int = 2, # thoi gian toi thieu de duoc coi la shaking
                 fps: int = 13,
                 merge_gap_sec: int = 5,
                 separate_gap_sec: int = 30,
                 min_shaking_time: int = 7):  # Thời gian tối thiểu để giữ lại segment
        """
        :param d_min: Ngưỡng max_distance
        :param s_min: Ngưỡng speed
        :param ratio_min: Ngưỡng speed / max_distance
        :param min_duration: Số giay liên tiếp để coi là shaking
        :param fps: Frames per second của video
        :param merge_gap_sec: Gộp 2 lần shake nếu cách nhau < Xs
        :param separate_gap_sec: Giữ tách biệt nếu cách nhau >= Xs
        :param min_shaking_time: Thời gian tối thiểu (giây) để giữ lại segment
        """
        self.d_min = d_min
        self.s_min = s_min
        self.d_max = d_max
        self.ratio_min = ratio_min
        self.min_duration = min_duration
        self.fps = fps
        self.merge_gap_frames = merge_gap_sec * fps
        self.separate_gap_frames = separate_gap_sec * fps
        self.min_shaking_time = min_shaking_time

    def detect_shaking(self, shaking_count: Dict[str, ShakingResult]) -> Dict[str, float]:
        """
        Detect shaking durations for all hands combined.
        :return: Dict[start_time, duration in seconds]
        """
        # Step 1: Detect raw shaking segments for all hands
        all_segments = []
        # print(shaking_count)
        for hand_id, result in shaking_count.items():
            actions = result.data
            current_start = None
            consecutive = 0 # don vi la giay(toi thieu bao nhieu s thi append)

            for action in actions:
                distance = action.distance
                max_dist = action.max_distance
                frame_id = action.frame_id
                
                if distance is None or max_dist is None:
                    current_start = None
                    consecutive = 0
                    continue

                ratio = distance / max_dist if max_dist > 0 else 0
                # print(ratio, hand_id, int(frame_id/13))
                if (max_dist >= self.d_min and
                        distance >= self.s_min and
                        ratio >= self.ratio_min):
                    consecutive += 1
                    
                    if current_start is None:
                        current_start = frame_id
                else:
                    if consecutive >= self.min_duration:
                        all_segments.append((current_start, frame_id - 1))
                        # print(current_start, frame_id - 1)

                    current_start = None
                    consecutive = 0

            # # Handle case where shaking continues until the end
            # if consecutive >= self.min_duration:
            #     all_segments.append((current_start, actions[-1].frame_id))
        
        # Sort segments by start frame
        all_segments.sort(key=lambda x: x[0])
        # print(all_segments)
        
        # Step 2: Merge close segments across all hands
        if not all_segments:
            return []
            
        merged_segments = []
        current_start, current_end = all_segments[0]
        
        for start, end in all_segments[1:]:
            if start - current_end <= self.merge_gap_frames:
                # Merge segments
                current_end = max(current_end, end)
            else:
                # Add current merged segment
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add the last segment
        merged_segments.append((current_start, current_end))
        
        # Step 3: Split segments based on separate_gap_sec
        final_segments = []
        
        for start, end in merged_segments:
            # Check if this merged segment has internal gaps >= separate_gap_sec
            # For simplicity, we'll keep merged segments as is since they were already
            # merged based on merge_gap_sec. The separate logic would need more complex
            # handling to track original gaps within merged segments.
            final_segments.append((start, end))
        
        # Step 4: Filter segments by minimum duration and convert to result format
        results = {}
        
        for start_frame, end_frame in final_segments:
            duration_sec = (end_frame - start_frame + 1) / self.fps
            
            # Only keep segments longer than min_shaking_time
            if duration_sec >= self.min_shaking_time:
                start_time_sec = start_frame / self.fps
                results[f"{start_time_sec:.2f}"] = round(duration_sec, 2)
        

        outputs = []
        for start_time in results.keys():
            outputs.append({'start_time': int(float(start_time)),
                            'duration': results[start_time]})

        return outputs

class Processor(object):
    def __init__(self, serial):
        self.fps = 13
        self.hand_manager = HandManager(max_history_frames=self.fps, inactive_frame_threshold=self.fps * 2)
        self.hand_mapper = HandPersonMapper()

        

        self.alcol_tracker = AlcoholBottleTracker()
        self.serial = serial
        self.alcohol_rois = {}

        # Khởi tạo cho sự kiện chạm chai cồn
        self.alcohols_contacts = {}
        self.touchs_events: Dict[str, TouchEvent] = {}   # Lưu các sự kiện chạm vào chai cồn
        self.min_overlap_frames = 3
        self.overlap_threshold = 0.05
        self.cooldowns = {}   # thay cho self.timeout → thời gian chờ sau mỗi event

        # Trạng thái theo dõi chạm tay
        self.is_touching = {}
        self.last_hand_id = {}

        # Khởi tạo cho washing event
        self.washing_count: Dict[str, WashingAction] = {}
        self.timeout_washing = {}

        # Khởi tạo cho shaking event
        self.shaking_count: Dict[int, ShakingResult] = {}
        self.shaking_interval: Dict[str, int] = {}

        self.initation = False
        self.bottle = {}

    def set_alcohol_rois(self, alcohol_results, frame_id):
        self.alcol_tracker.process_frame(alcohol_results, frame_id)
        rois = self.alcol_tracker.get_status()
        if not len(rois):
            return False
        
        if not self.initation:
            # self.is_touching = {roi_idx: False for roi_idx in range(len(alcohol_rois))}
            for bottle in rois:
                self.alcohols_contacts[bottle['track_id']] = deque(maxlen=self.min_overlap_frames)
                self.is_touching[bottle['track_id']] = False

            self.num_bottle = len(rois)
            
            self.initation = True


        self.alcohol_rois = {}
        for bottle in rois:
            self.alcohol_rois[bottle['track_id']] = bottle['bbox']

            if bottle['track_id'] not in self.alcohols_contacts.keys():
                self.alcohols_contacts[bottle['track_id']] = deque(maxlen=self.min_overlap_frames)
                self.is_touching[bottle['track_id']] = False

        return True

    def calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                         bbox2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_intersect = max(x1_1, x1_2)
        y1_intersect = max(y1_1, y1_2)
        x2_intersect = min(x2_1, x2_2)
        y2_intersect = min(y2_1, y2_2)

        if x1_intersect >= x2_intersect or y1_intersect >= y2_intersect:
            return 0.0

        intersect_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersect_area

        return intersect_area / union_area if union_area > 0 else 0.0

    def check_alcohol_contact(self, hand : HandTrack):
        contacted_rois = []
        for roi_idx, roi in self.alcohol_rois.items():
            overlap = self.calculate_overlap(hand.bbox, roi)
            if overlap >= self.overlap_threshold:
                # print("roi_idx", roi_idx)
                contacted_rois.append(roi_idx)
                # print(hand.track_id)
        return contacted_rois

    def update_alcohol_contact(self, frame_id):
        for (hand_id, hand) in self.hand_manager.current_hands.items():
            contacted_rois = self.check_alcohol_contact(hand)
            # print("contacted_rois", contacted_rois)
            for roi_idx in contacted_rois:
                self.alcohols_contacts[roi_idx].append((frame_id, hand.person_id))

    def check_valid_alcohol(self, frame_id: int) -> List[TouchEvent]:
        # print(frame_id, self.hand_manager.current_hands)
        touchs_event = []

        for roi_idx, contact_frames in self.alcohols_contacts.items():
            if roi_idx not in self.alcohol_rois:
                continue

            # Nếu trước đó đang chạm mà bây giờ không đủ frame → coi là rời tay
            if self.is_touching[roi_idx]:
                # print(contact_frames)
                # print(roi_idx)
                n_leave_frame = frame_id - contact_frames[-1][0]
                if n_leave_frame > self.min_overlap_frames + 2:
                    # hand_id = self.last_hand_id.get(roi_idx, None)
                    person_id = contact_frames[-1][1]
                    
                    if person_id:
                        # print(person_id, self.hand_manager.current_hands)
                        touch_event = TouchEvent(
                            roi_idx=roi_idx,
                            person_id=person_id,
                            frame_idx=frame_id,  # đánh dấu lúc rời tay
                            time_out=self.fps * 5
                        )
                        # reset trạng thái
                        print(f"person id {person_id} touched in second", int(frame_id/13))
                        touchs_event.append(touch_event)

                    self.is_touching[roi_idx] = False
                    # self.cooldowns[roi_idx] = self.fps * 2
                    self.alcohols_contacts[roi_idx] = deque(maxlen=self.min_overlap_frames)
                continue
            
            # Nếu ROI đang trong cooldown thì bỏ qua
            if roi_idx in self.cooldowns.keys():
                self.alcohols_contacts[roi_idx].clear()
                if self.cooldowns[roi_idx] > 0:
                    self.cooldowns[roi_idx] -= 1
                    continue
                else:
                    self.cooldowns.pop(roi_idx)
            # Kiểm tra có đủ số frame liên tục không
            # print(roi_idx, contact_frames)
            consecutive_count = 1
            for i in range(1, len(contact_frames)):
                if contact_frames[i][0] - contact_frames[i-1][0] <= 2:
                    consecutive_count += 1
                else:
                    consecutive_count = 1

            # Nếu đủ số frame liên tục và chưa set touching → set flag
            if consecutive_count >= self.min_overlap_frames and not self.is_touching[roi_idx]:
                self.is_touching[roi_idx] = True
                self.last_hand_id[roi_idx] = contact_frames[-1][1]
                self.cooldowns[roi_idx] = self.fps * 1

        return touchs_event


    def washing_check(self, person_id, frame_id, touch_id):
        if touch_id not in self.washing_count.keys():
            self.washing_count[touch_id] = WashingAction(frame_id)

        # print(self.washing_count[touch_id])
        hands_id_washing = self.hand_manager.get_hands_with_personid(person_id)
        if len(hands_id_washing) == 1:
            self.washing_count[touch_id].one_hand.append(frame_id)
            # print('one hand')
            return

        if len(hands_id_washing) > 2:
            self.washing_count[touch_id].overlap.append(frame_id)
            # print("overlap  3 hand")
            return
        

        # if len(hands_id_washing) != 2 and len(hands_id_washing) != 1:
        #     print("return")
        #     return
        if len(hands_id_washing) == 2:
            v1 = self.hand_manager.get_hand_velocity(hands_id_washing[0], round(self.fps))
            v2 = self.hand_manager.get_hand_velocity(hands_id_washing[1], round(self.fps))
            if v1 is None or v2 is None:
                # print('none')
                return

            overlap_hand = self.calculate_overlap(self.hand_manager.get_hand_track(hands_id_washing[0]).bbox, 
                                                self.hand_manager.get_hand_track(hands_id_washing[1]).bbox)

            if overlap_hand > 0.2:
                self.washing_count[touch_id].overlap.append(frame_id)
                # print("overlap")
                return
        
        # average_v = (v1 + v2) / 2
        # if average_v > 6 and (v1 > 5 and v2 > 5):
        #     self.washing_count[touch_id].append(frame_id)
        #     return
    
    def shaking_check(self, hand_id, frame_id, touch_id = None):
        if hand_id not in self.shaking_count.keys():
            self.shaking_count[hand_id] = ShakingResult()

        distance = self.hand_manager.get_hand_distance_on_frame(hand_id, window_size=round(self.fps))
        max_distance = self.hand_manager.get_max_distance_on_time(hand_id, window_size=round(self.fps))
        # print(hand_id, distance, max_distance)
        shaking_action = ShakingAction(distance=distance, 
                                       max_distance=max_distance,
                                       frame_id=frame_id)
        
        self.shaking_count[hand_id].data.append(shaking_action)
        
    def get_max_frequency_person(self):
        person_ids = [self.touchs_events[tid].person_id for tid in self.touchs_events]
        return person_ids[-1]

        if not person_ids:  # trường hợp rỗng
            return None

        person_count = Counter(person_ids)
        max_count = max(person_count.values())

        # Lấy tất cả person_id có tần suất bằng max_count
        top_persons = [pid for pid, cnt in person_count.items() if cnt == max_count]

        if len(top_persons) == 1:
            return top_persons[0]
        else:
            # tìm person_id cuối cùng trong chuỗi person_ids mà có trong top_persons
            for pid in reversed(person_ids):
                if pid in top_persons:
                    return pid

    def process_frame(self, hand_results, pose_results, alcohol_results, frame_id):
        """Process frame by frame"""
        # detect -> tracking -> mapping (hand <-> person_id)
        hands_track, pose_results = self.hand_mapper.process_frame(hand_results=hand_results,
                                                                   pose_results=pose_results,
                                                                   frame_id=frame_id)

        # detect alcohol -> tracking -> update
        is_alcol = self.set_alcohol_rois(alcohol_results, frame_id)
        
        # update hand to hand manager
        self.hand_manager.update_frame(hands_track, frame_id)
        self.hand_manager.cleanup_inactive_hands(frame_id)

        # check alcohol contact
        self.update_alcohol_contact(frame_id)
        touchs = self.check_valid_alcohol(frame_id)

        if len(touchs) > 0:
            # Has touch event -> running check washing
            for touch_event in touchs:
                # set timeout/add to queue for touching/washing event
                self.timeout_washing[touch_event.id] = self.fps * 5
                self.touchs_events[touch_event.id] = touch_event

        # washing check: mỗi lần có sự kiện chạm cồn thì bắt đầu check.
        for touch_id, touch_event in self.touchs_events.items():
            if self.timeout_washing[touch_id] > 0:
                # debug: hehe
                self.washing_check(touch_event.person_id, frame_id, touch_event.id)
                self.timeout_washing[touch_id] -= 1

        # shaking check: lấy personID đầu tiên chạm vào chai cồn làm chuẩn. -> check từ đó.
        if len(self.touchs_events.keys()) > 0:
            # key = list(self.touchs_events.keys())[-1]
            
            person_id = self.get_max_frequency_person()
            # print(person_id)
            hands_id_shaking = self.hand_manager.get_hands_with_personid(person_id)
            # print(person_id)
            for hand_id in hands_id_shaking:
                if hand_id not in self.shaking_interval.keys():
                    self.shaking_interval[hand_id] = round(self.fps)
            
            # print(self.shaking_interval.keys())

            for hand_id in self.shaking_interval.keys():
                if self.shaking_interval[hand_id] == 0:
                    self.shaking_check(hand_id, frame_id)
                    self.shaking_interval[hand_id] = round(self.fps)

                self.shaking_interval[hand_id] -= 1
    
    def postprocess(self, fps):
        results = []
        for touch_id, washing_action in self.washing_count.items():
            result = {}
            one_hand_count = washing_action.get_one_hand()
            overlap_count = washing_action.get_overlap()
            print(f"Frame ID: {int(touch_id.split('_')[-1]):04d}",f"| Overlap: {overlap_count:02d}", f"| One Hand: {one_hand_count:02d}")
            total_time = max((one_hand_count + overlap_count + self.min_overlap_frames + 2)/fps, 1)

            start_frame_id = washing_action.frame_id - self.min_overlap_frames - 2

            is_action_valid = (
                (overlap_count >= 4 and one_hand_count >= 4) 
                or (overlap_count >= 2 and one_hand_count >= 14) 
                or (overlap_count >= 6 and one_hand_count >= 2) 
                or (overlap_count >= 8 and one_hand_count >= 2) 
                or (overlap_count >= 10 and one_hand_count >= 1) 
                or (overlap_count >= 16 and one_hand_count >= 0) 
                or (overlap_count >= 1 and one_hand_count >= 26)
            )

            if is_action_valid:
                # Calculate the start time in seconds
                start_time_seconds = start_frame_id / fps
                
                # Populate the result dictionary
                result["start_time"] = start_time_seconds
                result["duration"] = total_time
                
                # Add the result to the list
                results.append(result)
        return results


class Runner:
    def __init__(self, serial, alcohol_paths, hand_paths, pose_paths, debug = False):
        self.processor = Processor(serial)
        self.shaking_detector = ShakingDetector()
        self.format = "%Y-%m-%dT%H:%M:%S.%fZ"

        self.expand_shaking = 1
        self.expand_washing = 1
        self.debug = debug

        self.alcohol_reader = BinFileReader(alcohol_paths)
        self.hand_reader = BinFileReader(hand_paths)
        self.pose_reader = BinFileReader(pose_paths)

        self.alcohol_frames = self.alcohol_reader.read()
        self.hand_frames = self.hand_reader.read()
        self.pose_frames = self.pose_reader.read()

        self.fake = False


    def run(self, start_video_time, end_video_time):
        actions = []
        process_frame = 0
        start_time_expand, end_time_expand = get_expand_timestamp(start_video_time, end_video_time)
        start_timestamp, end_timestamp = get_timestamp(start_video_time, end_video_time)
        

        for (hand_frame, pose_frame, alcohol_frame) in zip(self.hand_frames, self.pose_frames, self.alcohol_frames):
            self.processor.process_frame(hand_results=hand_frame.detections,
                                         pose_results=pose_frame.detections,
                                         alcohol_results=alcohol_frame.detections,
                                         frame_id=alcohol_frame.frame_index)
        
        shaking_data = self.processor.shaking_count
        shaking_results = self.shaking_detector.detect_shaking(shaking_data)

        for action in shaking_results:
            duration = action['duration']
            start_time = action['start_time'] + start_time_expand

            if start_time - start_time_expand < 30:
                if start_time + duration - (start_timestamp) > 13:
                    if self.fake:
                        start_timestamp += random.randint(-20, 20)
                    start_datetime = datetime.fromtimestamp(start_timestamp, hanoi_tz).isoformat(timespec="seconds")

                    action_dict = {"action_id": "1",
                            "action_name": "Lắc vaccine",
                            "start_time": start_datetime,
                            "duration": start_time + duration - start_timestamp + self.expand_shaking}

                    actions.append(action_dict)
                    continue 
                continue

            # if start_time < start_timestamp or start_time > end_timestamp:
            #     continue

            if self.fake:
                start_time += random.randint(-20, 20)
            start_datetime = datetime.fromtimestamp(start_time, hanoi_tz).isoformat(timespec="seconds")

            if duration < 7:
                continue

            action_dict = {"action_id": "1",
                        "action_name": "Lắc vaccine",
                        "start_time": start_datetime,
                        "duration": duration + self.expand_shaking}
            
            actions.append(action_dict)

        washing_results = self.processor.postprocess(13)
        # print(washing_results, start_timestamp, start_time_expand)
        for action in washing_results:
            start_time = action['start_time'] + start_time_expand
            duration = action['duration']

            if (start_time < start_timestamp) or start_time > end_timestamp:
                continue

            if self.fake:
                start_time += random.randint(-20, 20)
            start_datetime = datetime.fromtimestamp(start_time, hanoi_tz).isoformat(timespec="seconds")

            duration = action['duration']
            action_dict = {"action_id": "2",
                            "action_name": "Sát khuẩn tay",
                            "start_time": start_datetime,
                            "duration": duration + self.expand_washing}
            actions.append(action_dict)

        return actions
    
if __name__ == "__main__":
    import os
    import json
    

    with open("run.log") as f:
        for line in f:
            if not line:
                break
            
            message = json.loads(line)
            raw_alarm = message['alarm']['raw_alarm']
            record_list = raw_alarm['record_list']

            video_start_time = raw_alarm['video_start_time']
            video_end_time = raw_alarm['video_end_time']
            print(video_start_time, video_end_time)
            serial = raw_alarm['serial']
            moment_times = [rec['moment_time'] for rec in raw_alarm['record_list']]
            durations = [rec['duration'] for rec in raw_alarm['record_list']]
            # video_paths = [rec['file'] for rec in raw_alarm['record_list']]

            alcohol_path = message['alcohol']
            hand_path = message['hand']
            pose_path = message['pose']
            
            runner = Runner(serial, alcohol_path, hand_path, pose_path)

            actions = runner.run(video_start_time, video_end_time)

            print(json.dumps(actions, indent=2))
            break
