import os
import cv2
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional
import requests
import json
import os
import pytz
import cv2
from datetime import datetime

from python_app.tracker.byte_tracker import BYTETracker
from python_app.utils.video_reader import VideoReader

pose_score = 0.2
hand_score = 0.2

@dataclass
class HandTrack:
    """Lưu trữ thông tin tracking của bàn tay"""
    track_id: str  # Changed to string format: "{left/right}_{person_id}"
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    confidence: float
    frame_id: float
    person_id: int
    hand_type: str  # 'left', 'right', or 'unknown'

class HandPersonMapper:
    def __init__(self):
        """
        Initialize the mapper with YOLOv11 models
        
        Args:
            pose_model_path: Path to YOLOv11 pose estimation model
            hand_model_path: Path to YOLOv11 detection model for hands
        """
        # self.pose_model = YOLOv8TensorRT(pose_model_path, 
        #                                  conf_threshold = 0.4, 
        #                                  iou_threshold = 0.55, is_pose=True)
        
        # self.hand_model = YOLOv8TensorRT(hand_model_path, 
        #                                  conf_threshold=hand_score, 
        #                                  iou_threshold=0.6)
        
        # COCO pose keypoint indices (YOLOv11 uses COCO format)
        self.LEFT_WRIST_IDX = 9
        self.RIGHT_WRIST_IDX = 10
        self.LEFT_ELBOW_IDX = 7
        self.RIGHT_ELBOW_IDX = 8
        self.LEFT_SHOULDER_IDX = 5
        self.RIGHT_SHOULDER_IDX = 6

        self.tracker = BYTETracker(config_path="/age_gender/detection/tracker/bytetrack.yaml")
        
        # Distance threshold for hand-wrist association (in pixels)
        self.distance_threshold = 100
        
        # Tracking state for consistent hand IDs
        self.hand_tracking_state = {}  # Format: {yolo_track_id: {person_id: int, hand_type: str}}
        # self.executor = ThreadPoolExecutor(max_workers=2)

        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_arm_trajectory(self, shoulder, elbow, wrist):
        """
        Calculate arm trajectory and predict hand region
        
        Args:
            shoulder: (x, y) shoulder position
            elbow: (x, y) elbow position  
            wrist: (x, y) wrist position
            
        Returns:
            dict: Contains arm vector, predicted hand region, arm length
        """
        if shoulder is None or elbow is None or wrist is None:
            return None
        
        # Calculate arm segments
        upper_arm_vec = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        forearm_vec = (wrist[0] - elbow[0], wrist[1] - elbow[1])
        
        # Calculate arm lengths
        upper_arm_len = math.sqrt(upper_arm_vec[0]**2 + upper_arm_vec[1]**2)
        forearm_len = math.sqrt(forearm_vec[0]**2 + forearm_vec[1]**2)
        
        # Predict hand region (extend from wrist along forearm direction)
        if forearm_len > 0:
            # Normalize forearm vector
            forearm_unit = (forearm_vec[0] / forearm_len, forearm_vec[1] / forearm_len)
            
            # Predict hand center (extend ~15% of forearm length from wrist)
            hand_extension = forearm_len * 0.15
            predicted_hand_center = (
                wrist[0] + forearm_unit[0] * hand_extension,
                wrist[1] + forearm_unit[1] * hand_extension
            )
            
            # Create hand search region (circle around predicted center)
            hand_search_radius = max(50, forearm_len * 0.3)  # Adaptive radius
            
            return {
                'predicted_center': predicted_hand_center,
                'search_radius': hand_search_radius,
                'arm_length': upper_arm_len + forearm_len,
                'forearm_vector': forearm_unit,
                'wrist_pos': wrist
            }
        
        return None
    
    def get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def find_best_hand_match(self, hand_center, persons_wrists):
        """
        Find the best matching person and hand type for a detected hand
        
        Args:
            hand_center: (x, y) center of detected hand
            persons_wrists: List of person wrist data
            
        Returns:
            tuple: (person_id, hand_type, confidence_score)
        """
        best_match = {'person_id': -1, 'hand_type': 'unknown', 'score': float('inf')}
        
        for person_data in persons_wrists:
            # Check each arm for this person
            for arm_data in person_data['arms']:
                arm_info = arm_data['info']
                arm_type = arm_data['type']  # 'left' or 'right'
                
                predicted_center = arm_info['predicted_center']
                search_radius = arm_info['search_radius']
                
                # Distance from hand to predicted hand position
                pred_distance = self.calculate_distance(hand_center, predicted_center)
                
                if pred_distance <= search_radius:
                    # Score based on how close to predicted center and wrist
                    wrist_distance = self.calculate_distance(hand_center, arm_info['wrist_pos'])
                    
                    # Combined score: weighted sum of prediction accuracy and wrist proximity
                    trajectory_score = (pred_distance / search_radius) * 0.6 + (wrist_distance / self.distance_threshold) * 0.4
                    
                    if trajectory_score < best_match['score']:
                        best_match = {
                            'person_id': person_data['person_id'],
                            'hand_type': arm_type,
                            'score': trajectory_score
                        }
            
            # Fallback to simple wrist distance if no trajectory match
            if best_match['score'] == float('inf'):
                for wrist_data in person_data['wrists']:
                    wrist_pos = wrist_data['position']
                    wrist_type = wrist_data['type']
                    distance = self.calculate_distance(hand_center, wrist_pos)
                    
                    if distance < self.distance_threshold:
                        simple_score = distance / self.distance_threshold + 1.0  # Add penalty for using fallback
                        if simple_score < best_match['score']:
                            best_match = {
                                'person_id': person_data['person_id'],
                                'hand_type': wrist_type,
                                'score': simple_score
                            }
        
        return best_match['person_id'], best_match['hand_type'], best_match['score']
    
    def update_hand_tracking_state(self, yolo_track_id, person_id, hand_type):
        """
        Update the tracking state for consistent hand identification
        
        Args:
            yolo_track_id: Original YOLO tracking ID
            person_id: Assigned person ID
            hand_type: 'left', 'right', or 'unknown'
        """
        if yolo_track_id not in self.hand_tracking_state:
            self.hand_tracking_state[yolo_track_id] = {
                'person_id': person_id,
                'hand_type': hand_type,
                'confidence': 1.0
            }
        else:
            # Update with temporal consistency
            current_state = self.hand_tracking_state[yolo_track_id]
            
            # If assignment is consistent, increase confidence
            if current_state['person_id'] == person_id and current_state['hand_type'] == hand_type:
                current_state['confidence'] = min(1.0, current_state['confidence'] + 0.1)
            else:
                # If assignment changed, decrease confidence
                current_state['confidence'] = max(0.0, current_state['confidence'] - 0.2)
                
                # If confidence drops too low, update the assignment
                if current_state['confidence'] < 0.3:
                    current_state['person_id'] = person_id
                    current_state['hand_type'] = hand_type
                    current_state['confidence'] = 0.5
    
    def generate_hand_track_id(self, person_id, hand_type):
        """
        Generate a consistent track ID based on person and hand type
        
        Args:
            person_id: Person ID
            hand_type: 'left', 'right', or 'unknown'
            
        Returns:
            str: Track ID in format "{hand_type}_{person_id}"
        """
        if person_id >= 0 and hand_type in ['left', 'right']:
            return f"{hand_type}_{person_id}"
        elif person_id >= 0:
            return f"unknown_{person_id}"
        else:
            return "unassigned"
    

    def process_frame(self, frame_id, hand_results, pose_results) -> List[HandTrack]:
        """
        Process a single frame and return list of HandTrack objects
        
        Args:
            frame: Input image/frame
            frame_id: Current frame ID
            
        Returns:
            List[HandTrack]: List of tracked hands with person assignments
        """
        current_timestamp = frame_id
        hand_tracks = []
        
        # Continue with tracking (phải chạy sau khi có pose_results)
        output_track = self.tracker.update(pose_results)

        # Tạo dict: key = chỉ số pose, value = track_id
        track_map = {track[-1]: track[4] for track in output_track}
        # Gán id nhanh qua dict lookup
        for pose_ind, pose in enumerate(pose_results):
            track_id = track_map.get(pose_ind)
            if track_id is not None:
                pose.id = track_id
        
        # Extract person wrist keypoints
        persons_wrists = []
        if len(pose_results) > 0:
            
            for person_idx, person_pose in enumerate(pose_results):
                kpts, confs, person_track_id = person_pose.keypoints.xy, person_pose.keypoints.conf, person_pose.id
                
                if person_track_id is None:
                    continue
                
                wrists = []
                arms_info = []
                
                # Left arm trajectory
                left_shoulder = (float(kpts[self.LEFT_SHOULDER_IDX][0]), float(kpts[self.LEFT_SHOULDER_IDX][1])) if confs[self.LEFT_SHOULDER_IDX] > pose_score else None
                left_elbow = (float(kpts[self.LEFT_ELBOW_IDX][0]), float(kpts[self.LEFT_ELBOW_IDX][1])) if confs[self.LEFT_ELBOW_IDX] > pose_score else None
                left_wrist = (float(kpts[self.LEFT_WRIST_IDX][0]), float(kpts[self.LEFT_WRIST_IDX][1])) if confs[self.LEFT_WRIST_IDX] > pose_score else None
                
                if left_wrist:
                    wrists.append({
                        'position': left_wrist,
                        'type': 'left'
                    })
                    
                    # Calculate left arm trajectory
                    left_arm_info = self.calculate_arm_trajectory(left_shoulder, left_elbow, left_wrist)
                    if left_arm_info:
                        arms_info.append({
                            'type': 'left',
                            'info': left_arm_info
                        })
                
                # Right arm trajectory  
                right_shoulder = (float(kpts[self.RIGHT_SHOULDER_IDX][0]), float(kpts[self.RIGHT_SHOULDER_IDX][1])) if confs[self.RIGHT_SHOULDER_IDX] > pose_score else None
                right_elbow = (float(kpts[self.RIGHT_ELBOW_IDX][0]), float(kpts[self.RIGHT_ELBOW_IDX][1])) if confs[self.RIGHT_ELBOW_IDX] > pose_score else None
                right_wrist = (float(kpts[self.RIGHT_WRIST_IDX][0]), float(kpts[self.RIGHT_WRIST_IDX][1])) if confs[self.RIGHT_WRIST_IDX] > pose_score else None
                
                if right_wrist:
                    wrists.append({
                        'position': right_wrist,
                        'type': 'right'
                    })
                    
                    # Calculate right arm trajectory
                    right_arm_info = self.calculate_arm_trajectory(right_shoulder, right_elbow, right_wrist)
                    if right_arm_info:
                        arms_info.append({
                            'type': 'right', 
                            'info': right_arm_info
                        })
                
                persons_wrists.append({
                    'person_id': int(person_track_id),
                    'wrists': wrists,
                    'arms': arms_info
                })
        
        # Extract hand detections using YOLOv11 tracking
        if len(hand_results) > 0:
            boxes = [x.bbox for x in hand_results]
            confidences = [x.conf for x in hand_results]
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                # Convert bbox to integers
                x1, y1, x2, y2 = map(int, box)
                hand_bbox = (x1, y1, x2, y2)
                hand_center = self.get_bbox_center(hand_bbox)
                
                # Find the best matching person and hand type
                person_id, hand_type, match_score = self.find_best_hand_match(hand_center, persons_wrists)
                
                # # Update tracking state for consistency
                # self.update_hand_tracking_state(yolo_track_id, person_id, hand_type)
                
                # # Use the tracked state for final assignment
                # tracked_state = self.hand_tracking_state[yolo_track_id]
                # final_person_id = tracked_state['person_id']
                # final_hand_type = tracked_state['hand_type']

                final_person_id = person_id
                final_hand_type = hand_type
                
                # Generate the new track ID format
                new_track_id = self.generate_hand_track_id(final_person_id, final_hand_type)
                
                # Create HandTrack object with enhanced information
                hand_track = HandTrack(
                    track_id=new_track_id,
                    bbox=hand_bbox,
                    center=hand_center,
                    confidence=float(conf),
                    frame_id=current_timestamp,
                    person_id=final_person_id,
                    hand_type=final_hand_type
                )
                
                hand_tracks.append(hand_track)
        
        return hand_tracks, pose_results
    
    def visualize_results(self, frame, hand_tracks: List[HandTrack], pose_results):
        """
        Visualize the tracking results on the frame
        
        Args:
            frame: Input frame
            hand_tracks: List of HandTrack objects
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw hand tracking results
        for hand_track in hand_tracks:
            x1, y1, x2, y2 = hand_track.bbox
            
            # Color based on hand type and person assignment
            if hand_track.person_id >= 0:
                # Use different colors for different persons
                base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                base_color = base_colors[hand_track.person_id % len(base_colors)]
                
                # Modify color based on hand type
                if hand_track.hand_type == 'left':
                    color = base_color  # Keep original color for left hand
                elif hand_track.hand_type == 'right':
                    # Make right hand color lighter
                    color = tuple(min(255, c + 50) for c in base_color)
                else:
                    # Make unknown hand type darker
                    color = tuple(max(0, c - 50) for c in base_color)
            else:
                color = (128, 128, 128)  # Gray for unassigned
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(annotated_frame, hand_track.center, 3, color, -1)
            
            # Add labels with hand type and person ID
            # label = f"{hand_track.track_id} ({hand_track.confidence:.2f})"
            label = f"{hand_track.track_id}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # # Add hand type indicator
            # if hand_track.hand_type in ['left', 'right']:
            #     type_label = hand_track.hand_type[0].upper()  # 'L' or 'R'
            #     cv2.putText(annotated_frame, type_label, (x1, y2+15), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw arm trajectories for debugging (optional)
        # pose_results = self.pose_model.track(frame, persist=True, conf=pose_score, verbose=False)
        if len(pose_results) > 0:
            keypoints = [x.keypoints.xy for x in pose_results]
            confidences = [x.keypoints.conf for x in pose_results]

            person_track_ids = [x.id for x in pose_results]
            person_track_bboxes = [x.bbox for x in pose_results]
            
            for kpts, confs, pid, bbox in zip(keypoints, confidences, person_track_ids, person_track_bboxes):
                cv2.putText(annotated_frame, str(pid), (int(bbox[0]), int(bbox[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

                # Draw arm trajectories
                for side, (shoulder_idx, elbow_idx, wrist_idx) in [
                    ('left', (self.LEFT_SHOULDER_IDX, self.LEFT_ELBOW_IDX, self.LEFT_WRIST_IDX)),
                    ('right', (self.RIGHT_SHOULDER_IDX, self.RIGHT_ELBOW_IDX, self.RIGHT_WRIST_IDX))
                ]:
                    
                    if (confs[shoulder_idx] > pose_score and confs[elbow_idx] > pose_score and confs[wrist_idx] > pose_score):
                        shoulder = (int(kpts[shoulder_idx][0]), int(kpts[shoulder_idx][1]))
                        elbow = (int(kpts[elbow_idx][0]), int(kpts[elbow_idx][1]))
                        wrist = (int(kpts[wrist_idx][0]), int(kpts[wrist_idx][1]))
                        
                        # Draw arm skeleton in light gray
                        cv2.line(annotated_frame, shoulder, elbow, (0, 255, 255), 4)
                        cv2.line(annotated_frame, elbow, wrist, (0, 255, 255), 4)
                        
                        # # Calculate and draw predicted hand region
                        # arm_info = self.calculate_arm_trajectory(shoulder, elbow, wrist)
                        # if arm_info:
                        #     pred_center = (int(arm_info['predicted_center'][0]), int(arm_info['predicted_center'][1]))
                        #     radius = int(arm_info['search_radius'])
                        #     cv2.circle(annotated_frame, pred_center, radius, (100, 100, 100), 1)
                        #     cv2.circle(annotated_frame, pred_center, 2, (150, 150, 150), -1)
        
        return annotated_frame
    
    def reset_tracking(self):
        """Reset the tracking state for both models"""
        # Reset YOLO tracking state
        self.pose_model.predictor = None
        self.hand_model.predictor = None
        
        # Reset hand tracking state
        self.hand_tracking_state.clear()
    
    def get_tracking_statistics(self):
        """Get statistics about current tracking state"""
        stats = {
            'total_tracked_hands': len(self.hand_tracking_state),
            'left_hands': 0,
            'right_hands': 0,
            'unknown_hands': 0,
            'assigned_hands': 0,
            'unassigned_hands': 0
        }
        
        for track_id, state in self.hand_tracking_state.items():
            if state['hand_type'] == 'left':
                stats['left_hands'] += 1
            elif state['hand_type'] == 'right':
                stats['right_hands'] += 1
            else:
                stats['unknown_hands'] += 1
                
            if state['person_id'] >= 0:
                stats['assigned_hands'] += 1
            else:
                stats['unassigned_hands'] += 1
        
        return stats

# Alternative function for batch processing
def process_video_file(video_path: str, output_path: str = None, alcohol_rois = [], start = None, end = None):
    """
    Process entire video file and return all hand tracks
    
    Args:
        video_path: Path to input video
        output_path: Optional path to save annotated video
        
    Returns:
        List of all HandTrack objects across all frames
    """
    mapper = HandPersonMapper(hand_model_path="/age_gender/detection/YOLO11/runs/detect/hand_new_datasets5/weights/best.pt")
    cap = cv2.VideoCapture(video_path)
    
    all_hand_tracks = []
    frame_count = 0
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # Fixed resolution: 720x480
        width, height = 720, 480
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count > start*13  and frame_count < end*13:
                
            hand_tracks, pose_result = mapper.process_frame(frame, frame_count)
            all_hand_tracks.extend(hand_tracks)
            
            if writer:
                annotated_frame = mapper.visualize_results(frame, hand_tracks, pose_result)
                # for roi in alcohol_rois:
                #     cv2.rectangle(annotated_frame, (roi[0], roi[1]), (roi[2], roi[3]), (120, 215,30), 2)

                # Resize frame to 720x480 before writing
                annotated_frame_resized = cv2.resize(annotated_frame, (720, 480))
                writer.write(annotated_frame_resized)
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    
    cap.release()
    if writer:
        writer.release()
    
    print(f"Total frames processed: {frame_count}")
    print(f"Total hand tracks: {len(all_hand_tracks)}")
    
    return all_hand_tracks



hanoi_tz = pytz.timezone("Asia/Ho_Chi_Minh")


def download_videos(url, video_path):
    # stream download để tránh load cả file vào RAM
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except:
        return False
    return True


def fix_isoformat(dt_str: str) -> str:
    if "." in dt_str:
        # bỏ phần microseconds, giữ lại timezone (nếu có)
        date_part, rest = dt_str.split(".", 1)
        if "+" in rest:
            _, tz = rest.split("+", 1)
            return f"{date_part}+{tz}"
        elif "-" in rest:
            _, tz = rest.split("-", 1)
            return f"{date_part}-{tz}"
        else:
            return date_part
    return dt_str

def get_timestamp(start: str, end: str):
    start = fix_isoformat(start)
    end   = fix_isoformat(end)

    start_time = datetime.fromisoformat(start).astimezone(hanoi_tz)
    end_time   = datetime.fromisoformat(end).astimezone(hanoi_tz)

    return start_time.timestamp(), end_time.timestamp()

def get_expand_timestamp(start: str, end: str):
    start = fix_isoformat(start)
    end   = fix_isoformat(end)
    start_time = datetime.fromisoformat(start).astimezone(hanoi_tz)
    end_time   = datetime.fromisoformat(end).astimezone(hanoi_tz)

    return start_time.timestamp() - 30 , end_time.timestamp() + 5

if __name__ == "__main__":
    import json
    # import pycuda.autoinit 

    mapper = HandPersonMapper(hand_model_path="/age_gender/detection/weights/hand/2025-10-30.engine",
                              pose_model_path="/age_gender/detection/weights/pose/2025-10-30.engine")
    # mapper = HandPersonMapper()
    
    with open("run.log") as f:
        for line in f:
            if not line:
                break

            message = json.loads(line)
            raw_alarm = message['alarm']['raw_alarm']
            record_list = raw_alarm['record_list']

            video_start_time = raw_alarm['video_start_time']
            video_end_time = raw_alarm['video_end_time']
            serial = raw_alarm['serial']
            moment_times = [rec['moment_time'] for rec in raw_alarm['record_list']]
            durations = [float(rec['duration']) for rec in raw_alarm['record_list']]
            video_paths = message['playback_location']

            os.makedirs(f"outputs/mapping/{serial}", exist_ok=True)
            output_path = f"outputs/mapping/{serial}/{os.path.basename(video_paths[0])}"
            # output_path = f"outputs/mapping/{datetime.now().strftime('%YY-%mm-%dd')}/mapping_{moment_times}.mp4"
            
            start_timestamp, end_timestamp = get_expand_timestamp(video_start_time, video_end_time)
            
            writer = None
            width, height = 1280, 720
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, 13, (width, height))
            
            video_reader = VideoReader(video_paths, moment_times, durations, serial)

            while True:
                ret, frame, frame_index, ts = video_reader.read_range(start_timestamp, end_timestamp)
                if not ret:
                    break

                hand_tracks, poses_result = mapper.process_frame(frame, frame_index)

                if writer:
                    annotated_frame = mapper.visualize_results(frame, hand_tracks, poses_result)
                    annotated_frame_resized = cv2.resize(annotated_frame, (width, height))
                    writer.write(annotated_frame_resized)
                    
            video_reader.release()
            writer.release()
            print(output_path)



            # for video_path, moment_time in zip(video_paths, moment_times):
            #     print(video_path)
            #     # vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            #     # fps = vr.get_avg_fps()

            #     cap = cv2.VideoCapture(video_path)
            #     if not cap.isOpened():
            #         raise RuntimeError(f"Cannot open video: {video_path}")

            #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
            #     fps = int(total_frames/900)

            #     frame_index = 0
            #     while frame_index < total_frames:
            #         ret, frame = cap.read()
            #         frame_index += 1  # luôn tăng theo metadata, kể cả frame lỗi
            #         if not ret:
            #             print("hong frame", frame_index)
            #             continue

            #         current_time = moment_time + frame_index / fps
            #         if current_time > end_timestamp:
            #             break

            #         if start_timestamp <= current_time <= end_timestamp:
            #             # cv2.putText(frame, f"{frame_index}", (50, 50), 5, 3, (255,0,0), 3)
            #             # writer.write(cv2.resize(frame, (720, 480)))
            #             # t1 = time.time()
            #             hand_tracks, pose_result = mapper.process_frame(frame, frame_index)
            #             # t2 = time.time()
            #             # print(t2 - t1)

            #             if writer:
            #                 annotated_frame = mapper.visualize_results(frame, hand_tracks, pose_result)
            #                 annotated_frame_resized = cv2.resize(annotated_frame, (width, height))
            #                 writer.write(annotated_frame_resized)
            #     cap.release()

            # writer.release()
            # print(output_path)
            

    