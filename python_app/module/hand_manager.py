import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

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

@dataclass
class HandHistory:
    """Lưu trữ lịch sử của một bàn tay"""
    track_id: int
    person_id: int
    positions: deque  # Lưu các center positions
    bboxes: deque     # Lưu các bounding boxes
    frame_ids: deque  # Lưu các frame_id
    confidences: deque # Lưu các confidence scores
    last_frame: int
    first_frame: int
    
    def __post_init__(self):
        if not isinstance(self.positions, deque):
            self.positions = deque(maxlen=100)  # Giới hạn 100 frames gần nhất
        if not isinstance(self.bboxes, deque):
            self.bboxes = deque(maxlen=100)
        if not isinstance(self.frame_ids, deque):
            self.frame_ids = deque(maxlen=100)
        if not isinstance(self.confidences, deque):
            self.confidences = deque(maxlen=100)

class HandManager:
    """Quản lý các bàn tay trong video để phục vụ action recognition"""
    
    def __init__(self, max_history_frames: int = 100, inactive_frame_threshold: int = 10):
        """
        Args:
            max_history_frames: Số frame tối đa lưu trong lịch sử
            inactive_frame_threshold: Số frame để coi một hand là inactive
        """
        self.max_history_frames = max_history_frames
        self.inactive_frame_threshold = inactive_frame_threshold
        
        # Lưu trữ hands hiện tại trong frame
        self.current_hands: Dict[str, HandTrack] = {}
        
        # Lưu trữ lịch sử của tất cả hands
        self.hand_histories: Dict[str, HandHistory] = {}
        
        # Thống kê
        self.current_frame_id = 0
        self.total_hands_detected = 0
        
        # Nhóm hands theo person
        self.person_hands: Dict[int, Set[int]] = defaultdict(set)
    
    def update_frame(self, hands: List[HandTrack], frame_id: int) -> None:
        """Cập nhật thông tin hands cho frame hiện tại"""
        self.current_frame_id = frame_id
        
        # Lưu hands hiện tại
        self.current_hands.clear()
        current_track_ids = set()
        
        for hand in hands:
            # Cập nhật frame_id nếu chưa có
            if hand.frame_id != frame_id:
                hand.frame_id = frame_id

            if hand.person_id == -1:
                continue

            self.current_hands[hand.track_id] = hand
            
            current_track_ids.add(hand.track_id)
            
            # Cập nhật hoặc tạo mới hand history
            if hand.track_id not in self.hand_histories:
                self.hand_histories[hand.track_id] = HandHistory(
                    track_id=hand.track_id,
                    person_id=hand.person_id,
                    positions=deque(maxlen=self.max_history_frames),
                    bboxes=deque(maxlen=self.max_history_frames),
                    frame_ids=deque(maxlen=self.max_history_frames),
                    confidences=deque(maxlen=self.max_history_frames),
                    first_frame=hand.frame_id,
                    last_frame=hand.frame_id
                )
                self.total_hands_detected += 1
            
            # Cập nhật history
            history = self.hand_histories[hand.track_id]
            history.positions.append(hand.center)
            history.bboxes.append(hand.bbox)
            history.frame_ids.append(hand.frame_id)
            history.confidences.append(hand.confidence)
            history.last_frame = hand.frame_id
            history.person_id = hand.person_id  # Cập nhật person_id
            
            # Cập nhật mapping person -> hands
            self.person_hands[hand.person_id].add(hand.track_id)
    
    def get_hand_track(self, track_id):
        return self.current_hands[track_id]
    
    def get_person_id(self, track_id):
        return self.current_hands[track_id].person_id
    
    def get_hands_with_personid(self, person_id: int):
        """Lấy danh sách các hand cùng 1 personID"""
        tracks_id = []
        for track_id, handtrack in self.current_hands.items():
            if handtrack.person_id == person_id:
                tracks_id.append(track_id)
        
        return tracks_id

    def get_current_hands(self) -> List[HandTrack]:
        """Lấy danh sách hands trong frame hiện tại"""
        return list(self.current_hands.values())
    
    def get_hands_by_person(self, person_id: int) -> List[HandTrack]:
        """Lấy tất cả hands của một person trong frame hiện tại"""
        return [hand for hand in self.current_hands.values() 
                if hand.person_id == person_id]
    
    def get_hand_trajectory(self, track_id: int, num_frames: int = None) -> List[Tuple[int, int]]:
        """Lấy quỹ đạo di chuyển của một hand"""
        if track_id not in self.hand_histories:
            return []
        
        history = self.hand_histories[track_id]
        positions = list(history.positions)
        
        if num_frames is None:
            return positions
        else:
            return positions[-num_frames:] if len(positions) >= num_frames else positions
    
    def distance(self, gap_x, gap_y):
        return math.hypot(gap_x, gap_y)

    def get_hand_velocity(self, track_id: int, window_size: int = 13) -> Optional[Tuple[float, float]]:
        """Tính vận tốc trung bình của hand trong window_size frames gần nhất (pixels/frame)"""
        if track_id not in self.hand_histories:
            return None
        
        history = self.hand_histories[track_id]
        positions = list(history.positions)
        frame_ids = list(history.frame_ids)
        
        if len(positions) < 2:
            return None
        
        # Lấy window_size frames gần nhất
        recent_positions = positions[-window_size:] if len(positions) >= window_size else positions
        recent_frame_ids = frame_ids[-window_size:] if len(frame_ids) >= window_size else frame_ids
        
        if len(recent_positions) < 2:
            return None
        
        # Tính vận tốc trung bình
        total_vx, total_vy = 0, 0
        count = 0
        
        for i in range(1, len(recent_positions)):
            df = recent_frame_ids[i] - recent_frame_ids[i-1]
            if df > 0:
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                total_vx += dx / df
                total_vy += dy / df
                count += 1
        
        if count == 0:
            return None
        
        velocity = self.distance(total_vx / count, total_vy / count)

        return velocity
    
    def get_max_distance_on_time(self, track_id, window_size = 13):
        if track_id not in self.hand_histories:
            return None
        
        history = self.hand_histories[track_id]
        positions = list(history.positions)
        # frame_ids = list(history.frame_ids)

        if len(positions) < 2:
            return None

        recent_positions = positions[-window_size:]
        # recent_frame_ids = frame_ids[-window_size:]

        def distance(p1, p2):
            return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        # Tìm khoảng cách lớn nhất
        max_distance = 0
        for i in range(len(recent_positions)):
            for j in range(i + 1, len(recent_positions)):
                d = distance(recent_positions[i], recent_positions[j])
                if d > max_distance:
                    max_distance = d
        
        return max_distance


    def get_hand_distance_on_frame(self, track_id: int, window_size: int = 13) -> Optional[float]:
        if track_id not in self.hand_histories:
            return 0

        history = self.hand_histories[track_id]
        positions = list(history.positions)
        frame_ids = list(history.frame_ids)

        if len(positions) < 2:
            return 0

        distance = 0.0

        recent_positions = positions[-window_size:]
        recent_frame_ids = frame_ids[-window_size:]

        total_dx, total_dy, total_df = 0, 0, 0

        for i in range(1, len(recent_positions)):
            df = recent_frame_ids[i] - recent_frame_ids[i - 1]

            if df > 0:
                dx = recent_positions[i][0] - recent_positions[i - 1][0]
                dy = recent_positions[i][1] - recent_positions[i - 1][1]
                avg_distance_per_frame = self.distance(dx, dy) / df

                if total_df + df > window_size:
                    remaining_frames = window_size - total_df
                    distance += avg_distance_per_frame * remaining_frames
                    total_df = window_size  # Đã đủ
                    break
                else:
                    distance += avg_distance_per_frame * df
                    total_df += df
            if total_df > window_size:
                break

        if total_df == 0:
            return 0

        return distance
    
    def get_hand_distance(self, track_id1: int, track_id2: int) -> Optional[float]:
        """Tính khoảng cách giữa hai hands trong frame hiện tại"""
        if track_id1 not in self.current_hands or track_id2 not in self.current_hands:
            return None
        
        hand1 = self.current_hands[track_id1]
        hand2 = self.current_hands[track_id2]
        
        dx = hand1.center[0] - hand2.center[0]
        dy = hand1.center[1] - hand2.center[1]
        
        return np.sqrt(dx*dx + dy*dy)
    
    def get_active_hands(self, current_frame_id: int = None) -> List[int]:
        """Lấy danh sách track_id của các hands đang active"""
        if current_frame_id is None:
            current_frame_id = self.current_frame_id
        
        active_hands = []
        for track_id, history in self.hand_histories.items():
            if current_frame_id - history.last_frame <= self.inactive_frame_threshold:
                active_hands.append(track_id)
        
        return active_hands
    
    def get_person_hand_interactions(self, person_id1: int, person_id2: int) -> Dict[str, float]:
        """Phân tích tương tác giữa hands của hai người"""
        hands1 = self.get_hands_by_person(person_id1)
        hands2 = self.get_hands_by_person(person_id2)
        
        if not hands1 or not hands2:
            return {"min_distance": float('inf'), "avg_distance": float('inf'), "num_interactions": 0}
        
        distances = []
        for hand1 in hands1:
            for hand2 in hands2:
                dist = self.get_hand_distance(hand1.track_id, hand2.track_id)
                if dist is not None:
                    distances.append(dist)
        
        if not distances:
            return {"min_distance": float('inf'), "avg_distance": float('inf'), "num_interactions": 0}
        
        return {
            "min_distance": min(distances),
            "avg_distance": sum(distances) / len(distances),
            "num_interactions": len(distances)
        }
    
    def cleanup_inactive_hands(self, current_frame_id: int = None) -> int:
        """Xóa các hands không hoạt động để tiết kiệm bộ nhớ"""
        if current_frame_id is None:
            current_frame_id = self.current_frame_id
        
        inactive_hands = []
        for track_id, history in self.hand_histories.items():
            if current_frame_id - history.last_frame > self.inactive_frame_threshold:
                inactive_hands.append(track_id)
        
        # Xóa khỏi histories
        for track_id in inactive_hands:
            del self.hand_histories[track_id]
        
        # Cập nhật person_hands mapping
        for person_id in self.person_hands:
            self.person_hands[person_id] -= set(inactive_hands)
        
        return len(inactive_hands)
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê tổng quan"""
        active_hands = self.get_active_hands(self.current_frame_id)
        
        return {
            "current_frame_id": self.current_frame_id,
            "total_hands_detected": self.total_hands_detected,
            "current_active_hands": len(active_hands),
            "total_tracked_hands": len(self.hand_histories),
            "unique_persons": len(self.person_hands),
            "hands_per_person": {pid: len(hands) for pid, hands in self.person_hands.items()}
        }
    
    def reset(self) -> None:
        """Reset tất cả dữ liệu"""
        self.current_hands.clear()
        self.hand_histories.clear()
        self.person_hands.clear()
        self.current_frame_id = 0
        self.total_hands_detected = 0