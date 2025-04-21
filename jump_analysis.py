import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import argparse
import math
import os
from datetime import datetime

class LongJumpAnalyzer:
    def __init__(self, video_path, athlete_height):
        """Initialize the Long Jump Analyzer.
       
        Args:
            video_path (str): Path to the long jump video
            athlete_height (float): Height of the athlete in meters
        """
        self.video_path = video_path
        self.athlete_height = athlete_height
        self.cap = cv2.VideoCapture(video_path)
       
        # Check if video opened successfully
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")
       
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
        # Initialize Mediapipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
       
        # Variables to store analysis data
        self.calibration_factor = None
        self.take_off_point = None
        self.landing_point = None
        self.take_off_angle = None
        self.jump_distance = None
        self.horizontal_velocity = None
        self.stride_lengths = []
        self.stride_frequencies = []
        self.foot_positions = []
        self.hip_positions = []
        self.frame_timestamps = []
        self.on_ground = True
        self.in_air = False
        self.has_landed = False
       
        # Setup output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"long_jump_analysis_{timestamp}.mp4"
        self.output_path = output_filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_filename, fourcc, self.fps,
                                  (self.frame_width, self.frame_height))
   
    def calibrate_space(self, standing_frame_index=0):
        """Calibrate pixel to real-world distance using athlete's height.
       
        Args:
            standing_frame_index (int): Frame index where athlete is standing upright
        """
        # Seek to the calibration frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, standing_frame_index)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read calibration frame")
       
        # Process the frame to get pose landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
       
        if not results.pose_landmarks:
            raise ValueError("No pose detected in calibration frame")
       
        # Get heel and head landmarks
        landmarks = results.pose_landmarks.landmark
        heel = (landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].x * self.frame_width,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].y * self.frame_height)
        head = (landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * self.frame_width,
                landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * self.frame_height)
       
        # Calculate pixel distance between heel and head
        pixel_height = math.dist(heel, head)
       
        # Calculate calibration factor (meters per pixel)
        self.calibration_factor = self.athlete_height / pixel_height
        print(f"Calibration factor: {self.calibration_factor:.5f} meters/pixel")
       
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
   
    def calculate_take_off_angle(self, hip, knee, ankle):
        """Calculate take-off angle based on hip, knee, and ankle positions.
       
        Args:
            hip (tuple): (x, y) coordinates of hip
            knee (tuple): (x, y) coordinates of knee
            ankle (tuple): (x, y) coordinates of ankle
           
        Returns:
            float: Take-off angle in degrees
        """
        # Calculate vectors
        vec1 = (hip[0] - knee[0], hip[1] - knee[1])
        vec2 = (ankle[0] - knee[0], ankle[1] - knee[1])
       
        # Calculate dot product
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
       
        # Calculate magnitudes
        mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
       
        # Calculate angle in radians and convert to degrees
        cos_angle = dot_product / (mag1 * mag2)
        # Clamp cos_angle to [-1, 1] to avoid math domain errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
       
        # For leg extension, we need to calculate 180 - angle
        extension_angle = 180 - angle_deg
       
        return extension_angle
   
    def detect_feet_on_ground(self, ankle_y, knee_y, hip_y, frame_index):
        """Detect if feet are on ground based on joint positions and velocity.
       
        Args:
            ankle_y (float): Y-coordinate of ankle
            knee_y (float): Y-coordinate of knee
            hip_y (float): Y-coordinate of hip
            frame_index (int): Current frame index
           
        Returns:
            bool: True if feet are on ground, False otherwise
        """
        # Store current foot position
        self.foot_positions.append((frame_index / self.fps, ankle_y))
        self.hip_positions.append((frame_index / self.fps, hip_y))
       
        # Need at least 5 frames to analyze movement
        if len(self.foot_positions) < 5:
            return self.on_ground
       
        # Check for vertical movement of ankle relative to hip
        last_5_foot_y = [pos[1] for pos in self.foot_positions[-5:]]
        last_5_hip_y = [pos[1] for pos in self.hip_positions[-5:]]
       
        # Calculate relative position (normalize by hip position)
        rel_foot_positions = [foot_y - hip_y for foot_y, hip_y in zip(last_5_foot_y, last_5_hip_y)]
       
        # Calculate velocity of foot (pixels per frame)
        foot_velocity = (rel_foot_positions[-1] - rel_foot_positions[0]) / 5
       
        # State transitions
        if self.on_ground and foot_velocity < -2:  # Negative velocity means moving up
            self.on_ground = False
            self.in_air = True
            self.take_off_point = (self.foot_positions[-5][0], ankle_y)
            return False
       
        if self.in_air and not self.has_landed:
            # Check if foot is going down and near bottom of frame
            if foot_velocity > 2 and ankle_y > self.frame_height * 0.85:
                self.in_air = False
                self.has_landed = True
                self.landing_point = (frame_index / self.fps, ankle_y)
                return True
            return False
       
        return self.on_ground
   
    def analyze_strides(self):
        """Analyze stride length and frequency from foot positions."""
        if len(self.foot_positions) < 10:
            print("Not enough foot position data to analyze strides")
            return
       
        # Only use data before take-off
        if self.take_off_point:
            take_off_time = self.take_off_point[0]
            pre_takeoff_positions = [pos for pos in self.foot_positions if pos[0] < take_off_time]
        else:
            pre_takeoff_positions = self.foot_positions
       
        # Detect foot strikes by looking for local minima in y-coordinate
        foot_strikes = []
        for i in range(1, len(pre_takeoff_positions) - 1):
            time, y = pre_takeoff_positions[i]
            prev_y = pre_takeoff_positions[i - 1][1]
            next_y = pre_takeoff_positions[i + 1][1]
           
            # Local minimum (foot strike)
            if y <= prev_y and y <= next_y:
                foot_strikes.append((time, y))
       
        # Calculate stride lengths and frequencies
        if len(foot_strikes) >= 2:
            for i in range(1, len(foot_strikes)):
                # Calculate time between strikes
                time_diff = foot_strikes[i][0] - foot_strikes[i-1][0]
               
                # Calculate distance between strikes (in pixels)
                position_diff = abs(foot_strikes[i][1] - foot_strikes[i-1][1])
               
                # Convert to real-world distance
                stride_length = position_diff * self.calibration_factor
               
                # Calculate frequency (strides per second)
                if time_diff > 0:
                    stride_frequency = 1 / time_diff
                else:
                    stride_frequency = 0
               
                self.stride_lengths.append(stride_length)
                self.stride_frequencies.append(stride_frequency)
       
        # Calculate horizontal velocity before take-off
        if self.stride_lengths and self.stride_frequencies:
            avg_stride_length = sum(self.stride_lengths) / len(self.stride_lengths)
            avg_stride_frequency = sum(self.stride_frequencies) / len(self.stride_frequencies)
            self.horizontal_velocity = avg_stride_length * avg_stride_frequency
   
    def calculate_jump_distance(self):
        """Calculate jump distance based on take-off and landing points."""
        if self.take_off_point is None or self.landing_point is None:
            print("Take-off or landing point not detected")
            return None
       
        # Calculate time in air
        time_in_air = self.landing_point[0] - self.take_off_point[0]
       
        # Calculate horizontal distance in pixels
        horizontal_pixels = self.frame_width * time_in_air * 0.3  # Approximate based on camera panning
       
        # Convert to real-world distance
        self.jump_distance = horizontal_pixels * self.calibration_factor
       
        return self.jump_distance
   
    def create_performance_graph(self):
        """Create performance graphs for stride length, frequency, and velocity."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
       
        # Plot stride length
        if self.stride_lengths:
            ax1.plot(range(len(self.stride_lengths)), self.stride_lengths, 'b-o', label='Stride Length (m)')
            ax1.set_title('Stride Length')
            ax1.set_xlabel('Stride Number')
            ax1.set_ylabel('Length (m)')
            ax1.grid(True)
       
        # Plot stride frequency
        if self.stride_frequencies:
            ax2.plot(range(len(self.stride_frequencies)), self.stride_frequencies, 'r-o', label='Stride Frequency (Hz)')
            ax2.set_title('Stride Frequency')
            ax2.set_xlabel('Stride Number')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.grid(True)
       
        fig.tight_layout()
       
        # Convert plot to opencv image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        graph_img = np.array(canvas.renderer.buffer_rgba())
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
       
        # Resize graph to fit in video
        graph_height = self.frame_height // 3
        graph_width = int(graph_img.shape[1] * (graph_height / graph_img.shape[0]))
        graph_img = cv2.resize(graph_img, (graph_width, graph_height))
       
        plt.close(fig)
        return graph_img
   
    def process_video(self):
        """Process the video and generate analysis."""
        frame_index = 0
        last_hip = None
        last_knee = None
        last_ankle = None
       
        # First pass to calibrate space
        self.calibrate_space()
       
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       
        # Process each frame
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
           
            # Convert BGR to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
            # Process frame with Mediapipe Pose
            results = self.pose.process(rgb_frame)
           
            # Create a copy for drawing
            annotated_frame = frame.copy()
           
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
               
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
               
                # Get hip, knee, and ankle positions
                hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * self.frame_width),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * self.frame_height))
               
                knee = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * self.frame_width),
                        int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * self.frame_height))
               
                ankle = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * self.frame_width),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * self.frame_height))
               
                # Store last valid joint positions
                if all(p > 0 for p in hip + knee + ankle):
                    last_hip = hip
                    last_knee = knee
                    last_ankle = ankle
               
                    # Detect if on ground or in air
                    is_on_ground = self.detect_feet_on_ground(ankle[1], knee[1], hip[1], frame_index)
                   
                    # Calculate take-off angle if just took off
                    if not is_on_ground and self.take_off_point and self.take_off_angle is None:
                        self.take_off_angle = self.calculate_take_off_angle(hip, knee, ankle)
                        print(f"Take-off angle: {self.take_off_angle:.2f} degrees")
               
                # Draw take-off point
                if self.take_off_point:
                    take_off_time = self.take_off_point[0]
                    cv2.circle(annotated_frame, (int(take_off_time * self.fps * 5), int(self.take_off_point[1])),
                               10, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, "Take-off", (int(take_off_time * self.fps * 5) - 50,
                                int(self.take_off_point[1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
               
                # Draw landing point
                if self.landing_point:
                    landing_time = self.landing_point[0]
                    cv2.circle(annotated_frame, (int(landing_time * self.fps * 5), int(self.landing_point[1])),
                               10, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "Landing", (int(landing_time * self.fps * 5) - 50,
                                int(self.landing_point[1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           
            # Add analysis overlay
            self.add_analysis_overlay(annotated_frame, frame_index)
           
            # Write frame to output video
            self.out.write(annotated_frame)
           
            # Update frame index
            frame_index += 1
            self.frame_timestamps.append(frame_index / self.fps)
           
            # Display progress
            if frame_index % 30 == 0:
                print(f"Processing: {frame_index}/{self.total_frames} frames ({frame_index/self.total_frames*100:.1f}%)")
       
        # Calculate jump distance
        if self.take_off_point and self.landing_point:
            self.calculate_jump_distance()
       
        # Analyze strides
        self.analyze_strides()
       
        # Create performance graph
        perf_graph = self.create_performance_graph()
       
        # Add final frame with summary
        self.create_summary_frame(perf_graph)
       
        # Release resources
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
       
        print(f"Analysis complete. Output saved to {self.output_path}")
        return self.output_path
   
    def add_analysis_overlay(self, frame, frame_index):
        """Add analysis overlay to the frame."""
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_index}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
       
        # Add calibration factor
        if self.calibration_factor:
            cv2.putText(frame, f"Calibration: {self.calibration_factor:.5f} m/px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
       
        # Add take-off angle
        if self.take_off_angle:
            cv2.putText(frame, f"Take-off angle: {self.take_off_angle:.2f}°", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       
        # Add jump distance
        if self.jump_distance:
            cv2.putText(frame, f"Jump distance: {self.jump_distance:.2f} m", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       
        # Add horizontal velocity
        if self.horizontal_velocity:
            cv2.putText(frame, f"Horizontal velocity: {self.horizontal_velocity:.2f} m/s", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       
        # Add phase indication
        if self.on_ground:
            phase = "Approach Run"
        elif self.in_air:
            phase = "Flight Phase"
        elif self.has_landed:
            phase = "Landed"
        else:
            phase = "Unknown"
       
        cv2.putText(frame, f"Phase: {phase}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
   
    def create_summary_frame(self, perf_graph):
        """Create a summary frame with performance metrics and graphs."""
        # Create blank frame
        summary_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
       
        # Add title
        cv2.putText(summary_frame, "LONG JUMP PERFORMANCE ANALYSIS",
                    (self.frame_width//2 - 300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
       
        # Add metrics
        metrics_y = 100
        metrics_x = 50
        line_height = 40
       
        cv2.putText(summary_frame, f"Athlete Height: {self.athlete_height:.2f} m",
                    (metrics_x, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
       
        if self.take_off_angle:
            cv2.putText(summary_frame, f"Take-off Angle: {self.take_off_angle:.2f}°",
                        (metrics_x, metrics_y + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
       
        if self.jump_distance:
            cv2.putText(summary_frame, f"Jump Distance: {self.jump_distance:.2f} m",
                        (metrics_x, metrics_y + 2*line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
       
        if self.horizontal_velocity:
            cv2.putText(summary_frame, f"Horizontal Velocity: {self.horizontal_velocity:.2f} m/s",
                        (metrics_x, metrics_y + 3*line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
       
        if self.stride_lengths:
            avg_stride = sum(self.stride_lengths) / len(self.stride_lengths)
            cv2.putText(summary_frame, f"Average Stride Length: {avg_stride:.2f} m",
                        (metrics_x, metrics_y + 4*line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
       
        if self.stride_frequencies:
            avg_freq = sum(self.stride_frequencies) / len(self.stride_frequencies)
            cv2.putText(summary_frame, f"Average Stride Frequency: {avg_freq:.2f} Hz",
                        (metrics_x, metrics_y + 5*line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
       
        # Add performance graph
        if perf_graph is not None:
            graph_y = metrics_y + 6*line_height
            summary_frame[graph_y:graph_y + perf_graph.shape[0],
                         (self.frame_width - perf_graph.shape[1])//2:
                         (self.frame_width - perf_graph.shape[1])//2 + perf_graph.shape[1]] = perf_graph
       
        # Add footer
        cv2.putText(summary_frame, "Analysis completed successfully",
                    (self.frame_width//2 - 200, self.frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
       
        # Write to output video
        for _ in range(self.fps * 5):  # Show summary for 5 seconds
            self.out.write(summary_frame)


def main():
    """Main function to parse arguments and run the analyzer."""
    parser = argparse.ArgumentParser(description='Long Jump Performance Analyzer')
    parser.add_argument('--video', type=str, required=True, help='Path to long jump video')
    parser.add_argument('--height', type=float, required=True, help='Athlete\'s height in meters')
   
    args = parser.parse_args()
   
    try:
        analyzer = LongJumpAnalyzer(args.video, args.height)
        output_path = analyzer.process_video()
        print(f"Analysis complete! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()