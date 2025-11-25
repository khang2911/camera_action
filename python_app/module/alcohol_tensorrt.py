import cv2
import math
import numpy as np
import pycuda.autoinit

from python_app.utils.utils import getConfig, unscale_box

alcohol_config = {"c03o24120001915": np.array([[193, 600, 270, 660]]),
                        #   "c03o24120001924": [777, 488, 1635, 1232],
                        #   "c03o24120001908" : [549, 651, 1369, 1249],
                        #   "c03o25010007839" : [837, 633, 2013, 1197],
                        #   "c03o25010007851" : [547, 700, 1487, 1235]
                          }
import numpy as np


class AlcoholBottleTracker:
    def __init__(self, fps=13, detect_duration=1.0, idle_duration=1.0, max_missing_cycles=10):
        """
        Khởi tạo tracker cho chai cồn
        
        Args:
            model_path: Đường dẫn đến model YOLOv11
            process_roi: [x1, y1, x2, y2] - Vùng ROI để detect
            fps: Số frame per second (mặc định 13)
            detect_duration: Thời gian detect (giây, mặc định 1s)
            idle_duration: Thời gian nghỉ (giây, mặc định 1s)
            max_missing_cycles: Số chu kỳ detect tối đa khi mất detection (mặc định 10)
        """
        # self.model = YOLOv8TensorRT(engine_path=model_path, conf_threshold=0.2, iou_threshold = 0.55)
        self.process_roi = None
        self.fps = fps
        self.detect_duration = detect_duration
        self.idle_duration = idle_duration
        self.max_missing_cycles = max_missing_cycles
        
        # Số frames cho mỗi giai đoạn
        self.frames_per_detect = int(fps * detect_duration)  # 13 frames
        self.frames_per_idle = int(fps * idle_duration)      # 13 frames
        self.frames_per_cycle = self.frames_per_detect + self.frames_per_idle  # 26 frames
        
        # Buffer để lưu detection results trong 1 chu kỳ detect
        self.detection_buffer = []
        
        # Tracking state
        self.current_tracks = {}  # {track_id: {"bbox": [x1,y1,x2,y2], "missing_cycles": 0}}
        self.next_track_id = 1
        
        # Lưu frame_id cuối cùng processed
        self.last_frame_id = -1
    
    def reset_tracking(self):
        self.process_roi = None

        # Buffer để lưu detection results trong 1 chu kỳ detect
        self.detection_buffer = []

        # Tracking state
        self.current_tracks = {}  # {track_id: {"bbox": [x1,y1,x2,y2], "missing_cycles": 0}}
        self.next_track_id = 1

        # Lưu frame_id cuối cùng processed
        self.last_frame_id = -1


    def _get_cycle_phase(self, frame_id):
        """
        Xác định phase hiện tại trong chu kỳ
        
        Returns:
            ("detect", frame_in_phase) hoặc ("idle", frame_in_phase)
        """
        position_in_cycle = frame_id % self.frames_per_cycle
        
        if position_in_cycle < self.frames_per_detect:
            return ("detect", position_in_cycle)
        else:
            return ("idle", position_in_cycle - self.frames_per_detect)
    
    def _is_end_of_detect_phase(self, frame_id):
        """Check xem có phải frame cuối của detect phase không"""
        phase, frame_in_phase = self._get_cycle_phase(frame_id)
        return phase == "detect" and frame_in_phase == self.frames_per_detect - 1
    
    def _calculate_iou(self, box1, box2):
        """Tính IoU giữa 2 bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Tính diện tích giao nhau
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Tính diện tích hợp nhau
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _merge_overlapping_boxes(self, boxes, iou_threshold=0.1):
        """Merge các boxes overlap > threshold"""
        if len(boxes) == 0:
            return []
        
        merged = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            
            current_group = [boxes[i]]
            used[i] = True
            
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                
                # Check overlap với bất kỳ box nào trong group
                should_merge = False
                for box in current_group:
                    if self._calculate_iou(box, boxes[j]) > iou_threshold:
                        should_merge = True
                        break
                
                if should_merge:
                    current_group.append(boxes[j])
                    used[j] = True
            
            # Lấy median của group
            merged.append(self._get_median_box(current_group))
        
        return merged
    
    def _get_median_box(self, boxes):
        """Lấy median box từ list boxes"""
        if len(boxes) == 1:
            return boxes[0]
        
        boxes = np.array(boxes)
        median_box = np.median(boxes, axis=0)
        return median_box.tolist()
    
    def _voting_boxes(self, buffer):
        """
        Voting từ buffer detection results
        Chọn ra các boxes xuất hiện nhiều nhất
        """
        if len(buffer) == 0:
            return []
        
        # Gộp tất cả boxes từ buffer
        all_boxes = []
        for detections in buffer:
            all_boxes.extend(detections)
        
        if len(all_boxes) == 0:
            return []
        
        # Merge các boxes overlap
        merged_boxes = self._merge_overlapping_boxes(all_boxes, iou_threshold=0.1)
        
        # Sort theo x1 (từ trái qua phải)
        merged_boxes.sort(key=lambda box: box[0])
        
        return merged_boxes
    
    def _assign_track_ids(self, detected_boxes):
        """
        Assign track IDs cho các detected boxes
        Ưu tiên giữ ID cũ nếu match được
        """
        if len(detected_boxes) == 0:
            # Tăng missing cycles cho tất cả tracks
            for tid in list(self.current_tracks.keys()):
                self.current_tracks[tid]["missing_cycles"] += 1
                if self.current_tracks[tid]["missing_cycles"] > self.max_missing_cycles:
                    del self.current_tracks[tid]
            return
        
        # Match detected boxes với current tracks
        matched_tracks = {}
        used_boxes = set()
        
        # Sort current tracks theo x1
        sorted_tracks = sorted(self.current_tracks.items(), 
                             key=lambda x: x[1]["bbox"][0])
        
        for tid, track_info in sorted_tracks:
            best_iou = 0
            best_box_idx = -1
            
            for i, box in enumerate(detected_boxes):
                if i in used_boxes:
                    continue
                
                iou = self._calculate_iou(track_info["bbox"], box)
                if iou > best_iou:
                    best_iou = iou
                    best_box_idx = i
            
            if best_iou > 0.1:  # Threshold để match
                matched_tracks[tid] = {
                    "bbox": detected_boxes[best_box_idx],
                    "missing_cycles": 0
                }
                used_boxes.add(best_box_idx)
        
        # Assign new IDs cho boxes chưa match
        new_boxes = [box for i, box in enumerate(detected_boxes) if i not in used_boxes]
        
        # Update current tracks
        self.current_tracks = matched_tracks
        
        # Add new tracks
        for box in new_boxes:
            self.current_tracks[self.next_track_id] = {
                "bbox": box,
                "missing_cycles": 0
            }
            self.next_track_id += 1
        
        # Reassign IDs theo thứ tự từ trái qua phải
        self._reassign_ids_by_position()
    
    def _reassign_ids_by_position(self):
        """
        Reassign IDs theo thứ tự từ trái qua phải
        ID luôn là 1, 2, 3, ...
        """
        if len(self.current_tracks) == 0:
            self.next_track_id = 1
            return
        
        # Sort tracks theo x1
        sorted_tracks = sorted(self.current_tracks.items(), 
                             key=lambda x: x[1]["bbox"][0])
        
        # Tạo mapping mới
        new_tracks = {}
        for new_id, (old_id, track_info) in enumerate(sorted_tracks, start=1):
            new_tracks[new_id] = track_info
        
        self.current_tracks = new_tracks
        self.next_track_id = len(new_tracks) + 1
    
    def _roi_to_frame_coords(self, roi_bbox):
        """
        Chuyển đổi bbox từ hệ tọa độ ROI sang hệ tọa độ frame gốc
        
        Args:
            roi_bbox: [x1, y1, x2, y2] trong hệ tọa độ ROI
        Returns:
            [x1, y1, x2, y2] trong hệ tọa độ frame gốc
        """
        roi_x1, roi_y1, roi_x2, roi_y2 = self.process_roi
        x1, y1, x2, y2 = roi_bbox
        
        return [
            int(x1 + roi_x1),
            int(y1 + roi_y1),
            int(x2 + roi_x1),
            int(y2 + roi_y1)
        ]
    
    def process_frame(self, result, frame_id):
        """
        Process frame
        
        Args:
            frame: Frame ảnh (numpy array)
            frame_id: ID của frame
        """

        self.process_roi = []
        # Kiểm tra xem có phải frame tiếp theo không
        if frame_id != self.last_frame_id + 1:
            # Reset nếu không liên tục
            self.detection_buffer = []
        
        self.last_frame_id = frame_id
        
        # Xác định phase hiện tại
        phase, frame_in_phase = self._get_cycle_phase(frame_id)
        
        if phase == "detect":
            # Trong phase detect: Cắt ROI và detect
            # roi_x1, roi_y1, roi_x2, roi_y2 = self.process_roi
            # roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # # Detect trên ROI
            # result = self.model.predict(roi_frame)
            
            # Lấy detections của class 0 (chai cồn)
            detections = []
            for i in range(len(result)):
                bbox = result[i].bbox
                detections.append(bbox)
            
            self.detection_buffer.append(detections)
            
            # Nếu là frame cuối của detect phase, thực hiện voting
            if self._is_end_of_detect_phase(frame_id):
                voted_boxes = self._voting_boxes(self.detection_buffer)
                self._assign_track_ids(voted_boxes)
                
                # Reset buffer
                self.detection_buffer = []
        
        # Trong phase idle: Không làm gì, chỉ giữ nguyên tracks
    
    def get_status(self):
        """
        Trả về status hiện tại của tracking với bbox trên hệ quy chiếu frame gốc
        
        Returns:
            List of dict: [{"track_id": tid, "bbox": [x1, y1, x2, y2]}]
        """
        status = []
        for tid, track_info in sorted(self.current_tracks.items()):
            # Chuyển đổi bbox từ ROI sang frame gốc
            # frame_bbox = self._roi_to_frame_coords(track_info["bbox"])
            status.append({
                "track_id": tid,
                "bbox": track_info["bbox"]
            })
        # print(status)
        return status
    
    def draw_tracks(self, frame, process_roi=None, color=(0, 255, 0)):
        """Vẽ các track lên frame"""
        if process_roi is not None:
            a1, b1, a2, b2 = process_roi
            cv2.rectangle(frame, (a1, b1), (a2, b2), color, 3)
        for obj in self.get_status():
            # print()
            tid = obj["track_id"]
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"ID {tid}",
                        (x1, y1 - 8),
                        1,
                        fontScale=2,
                        color=color,
                        thickness=2)
        return frame


def roi_check(video_path, roi, cam_id, model: AlcoholBottleTracker):
    _, image = cv2.VideoCapture(video_path).read()
    m1, n1, x2,y2 = roi
    frame = image[n1:y2, m1:x2]
    result = model.model.predict(frame)
    # print(result)
    # results = self.model(roi_frame, verbose=False, iou = 0.55)
    cv2.rectangle(image, (m1, n1), (x2, y2), color=(0, 255, 255), thickness=2)

    for i in range(len(result)):
        bbox = result[i]['bbox']
        conf = result[i]['confidence']
        x1,y1,x2,y2 = [int(x) for x in bbox]
        
        cv2.rectangle(image, (x1 + m1 ,y1 + n1), (x2 + m1, y2 + n1), color=(255, 0, 0), thickness=2)
        cv2.putText(image, str(round(conf, 2)), (x1 + m1 , y1 + n1-10) , 2, 0.7, (255, 0,0), 2)

    cv2.imwrite(f"roi_{cam_id}.jpg", image)

# if __name__ == "__main__":
#     # a = [693, 448, 1617, 1040]
#     # c = [828, 436, 1809, 850]
#     import json
#     import time
#     import os
#     # from hand_manager_dev_check import get_timestamp

#     with open("run.log") as f:
#         for line in f:
#             if not line:
#                 break
            
#             fps = 13
#             # Fixed resolution: 720x480
#             message = json.loads(line)
#             raw_alarm = json.loads(line)['alarm']['raw_alarm']
#             record_list = raw_alarm['record_list']

#             video_start_time = raw_alarm['video_start_time']
#             video_end_time = raw_alarm['video_end_time']
#             serial = raw_alarm['serial']
#             moment_times = [rec['moment_time'] for rec in raw_alarm['record_list']]
#             durations = [rec['duration'] for rec in raw_alarm['record_list']]
#             video_paths = message['playback_location']
            
#             image_w, image_h = message['video_resolution']['width'], message['video_resolution']['height']
            
#             cam_config = getConfig(serial)

#             if "box" not in cam_config:
#                 print("not config")
            
#             try:
#                 bbox_config = [cam_config['box'][0][0], cam_config['box'][0][1], cam_config['box'][2][0], cam_config['box'][2][1]]
#                 process_roi = unscale_box(bbox_config, image_w, image_h)
#             except:
#                 print("not config")

#             print(process_roi)

#             # start_timestamp, end_timestamp = get_timestamp(video_start_time, video_end_time)
#             # /age_gender/detection/outputs/alcohol_check
#             output_path = f"outputs/alcohol_check/{video_start_time.split('.')[0].replace(':', '_')}.mp4"
#             writer = None

#             if output_path:
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 # 13 = int(cap.get(cv2.CAP_PROP_FPS))
#                 # Fixed resolution: 720x480
#                 width, height = 1280, 720
#                 writer = cv2.VideoWriter(output_path, fourcc, 13, (width, height))

#             model = AlcoholBottleTracker(model_path="/age_gender/detection/weights/alcohol/2025-11-11_batch_1_640x640_fp16.engine")
            
#             # roi_check(video_paths[0], roi_config[serial], serial, model)

#             x = 0
#             for i, (video_path, moment_time) in enumerate(zip(video_paths, moment_times)):

#                 cap = cv2.VideoCapture(video_path)
#                 if not cap.isOpened():
#                     raise RuntimeError(f"Cannot open video: {video_path}")

#                 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 # fps = cap.get(cv2.CAP_PROP_FPS)
#                 fps = total_frames/float(durations[i])

#                 frame_index = 0
#                 while frame_index < total_frames:
#                     ret, frame = cap.read()
#                     frame_index += 1  # luôn tăng theo metadata, kể cả frame lỗi
#                     if not ret:
#                         print("hong frame", frame_index)
#                         continue

#                     current_time = moment_time + frame_index / fps
#                     # if current_time > end_timestamp:
#                     #     break

#                     # if start_timestamp <= current_time <= end_timestamp:
#                     #     x+=1
#                         # print(int(x)/13)
#                         # t1 = time.time()

#                         model.process_frame(frame, x, process_roi, serial=serial)
#                         t2 = time.time()
#                         # print(t2 - t1)
#                         if writer:
#                             annotated_frame_resized = model.draw_tracks(frame, process_roi)
#                             # cv2.imwrite("check_alcohol_tracking.png", annotated_frame_resized)
#                             annotated_frame_resized = cv2.resize(annotated_frame_resized, (1280, 720))
#                             writer.write(annotated_frame_resized)

#                 cap.release()
#             writer.release()
#             print(output_path)

