import cv2
import numpy as np
import pandas as pd
import os
import sys
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionTester:
    """Class for testing face recognition models"""
    
    def __init__(self, model_path="TrainingImageLabel/trainer.yml", student_details_path="StudentDetails/StudentDetails.csv"):
        self.model_path = Path(model_path)
        self.student_details_path = Path(student_details_path)
        self.recognizer = None
        self.face_cascade = None
        self.student_details = {}
        
        # Initialize components
        self._load_model()
        self._load_face_detector()
        self._load_student_details()
    
    def _load_model(self):
        """Load the trained face recognition model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(str(self.model_path))
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _load_face_detector(self):
        """Load the face detection cascade"""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face detection cascade")
        
        logger.info("Face detector loaded successfully")
    
    def _load_student_details(self):
        """Load student details from CSV file"""
        if self.student_details_path.exists():
            try:
                df = pd.read_csv(self.student_details_path)
                self.student_details = dict(zip(df['Enrollment'], df['Name']))
                logger.info(f"Loaded details for {len(self.student_details)} students")
            except Exception as e:
                logger.warning(f"Failed to load student details: {str(e)}")
                self.student_details = {}
        else:
            logger.warning(f"Student details file not found: {self.student_details_path}")
            self.student_details = {}
    
    def recognize_face(self, face_roi):
        """Recognize a single face ROI"""
        if self.recognizer is None:
            raise RuntimeError("Model not loaded")
        
        # Resize face for consistent recognition
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Predict using the trained model
        student_id, confidence = self.recognizer.predict(face_roi)
        
        return student_id, confidence
    
    def test_realtime(self, confidence_threshold=70, max_duration=None):
        """Test face recognition in real-time using webcam"""
        logger.info("Starting real-time face recognition test")
        logger.info("Press 'q' to quit, 's' to save screenshot, 'r' to reset recognition")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Recognition tracking
        recognition_history = {}
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    face_roi = gray[y:y+h, x:x+w]
                    
                    try:
                        # Recognize face
                        student_id, confidence = self.recognize_face(face_roi)
                        
                        # Determine if recognition is confident enough
                        if confidence < confidence_threshold:
                            # Get student name
                            student_name = self.student_details.get(student_id, f"ID_{student_id}")
                            
                            # Update recognition history
                            if student_id not in recognition_history:
                                recognition_history[student_id] = {
                                    'name': student_name,
                                    'count': 0,
                                    'avg_confidence': 0,
                                    'last_seen': time.time()
                                }
                            
                            history = recognition_history[student_id]
                            history['count'] += 1
                            history['avg_confidence'] = ((history['avg_confidence'] * (history['count'] - 1)) + confidence) / history['count']
                            history['last_seen'] = time.time()
                            
                            # Draw recognition result
                            color = (0, 255, 0)  # Green for recognized
                            label = f"{student_name} ({confidence:.1f})"
                            
                        else:
                            # Unknown face
                            color = (0, 0, 255)  # Red for unknown
                            label = f"Unknown ({confidence:.1f})"
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                    except Exception as e:
                        logger.error(f"Recognition error: {str(e)}")
                        # Draw error indicator
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, "Error", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Add status information
                status_y = 30
                cv2.putText(frame, f"Frame: {frame_count}", (10, status_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                status_y += 20
                cv2.putText(frame, f"Recognized: {len(recognition_history)}", (10, status_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if max_duration:
                    elapsed = time.time() - start_time
                    remaining = max_duration - elapsed
                    if remaining <= 0:
                        break
                    status_y += 20
                    cv2.putText(frame, f"Time: {remaining:.1f}s", (10, status_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Face Recognition Test - Press q to quit, s to save, r to reset', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"test_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    # Reset recognition history
                    recognition_history.clear()
                    logger.info("Recognition history reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Display test results
        self._display_test_results(recognition_history, frame_count, time.time() - start_time)
    
    def _display_test_results(self, recognition_history, frame_count, duration):
        """Display test results summary"""
        print(f"\n{'='*60}")
        print("FACE RECOGNITION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Test duration: {duration:.1f} seconds")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {frame_count/duration:.1f}")
        print(f"Unique faces recognized: {len(recognition_history)}")
        
        if recognition_history:
            print(f"\n{'Recognition Details:'}")
            print(f"{'ID':<8} {'Name':<20} {'Count':<8} {'Avg Conf':<10} {'Last Seen'}")
            print("-" * 60)
            
            for student_id, data in recognition_history.items():
                last_seen = time.time() - data['last_seen']
                print(f"{student_id:<8} {data['name']:<20} {data['count']:<8} "
                      f"{data['avg_confidence']:<10.1f} {last_seen:.1f}s ago")
        
        print(f"{'='*60}")
    
    def test_image(self, image_path, confidence_threshold=70):
        """Test face recognition on a single image"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Recognize face
            student_id, confidence = self.recognize_face(face_roi)
            
            # Get student name
            student_name = self.student_details.get(student_id, f"ID_{student_id}")
            
            # Determine if recognition is confident
            recognized = confidence < confidence_threshold
            
            result = {
                'face_index': i + 1,
                'student_id': student_id,
                'student_name': student_name,
                'confidence': confidence,
                'recognized': recognized,
                'bbox': (x, y, w, h)
            }
            results.append(result)
            
            # Draw result on image
            color = (0, 255, 0) if recognized else (0, 0, 255)
            label = f"{student_name} ({confidence:.1f})" if recognized else f"Unknown ({confidence:.1f})"
            
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display results
        print(f"\nTest results for {image_path.name}:")
        print(f"Faces detected: {len(faces)}")
        
        for result in results:
            status = "RECOGNIZED" if result['recognized'] else "UNKNOWN"
            print(f"  Face {result['face_index']}: {result['student_name']} "
                  f"(confidence: {result['confidence']:.1f}) - {status}")
        
        # Show image
        cv2.imshow(f'Test Result - {image_path.name}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return results

def main():
    """Main function for testing"""
    try:
        tester = FaceRecognitionTester()
        
        print("Face Recognition Testing Tool")
        print("=" * 40)
        print("1. Real-time testing (webcam)")
        print("2. Single image testing")
        print("3. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                duration = input("Enter test duration in seconds (or press Enter for unlimited): ").strip()
                max_duration = float(duration) if duration else None
                
                confidence = input("Enter confidence threshold (default 70): ").strip()
                confidence_threshold = float(confidence) if confidence else 70
                
                tester.test_realtime(confidence_threshold, max_duration)
                
            elif choice == '2':
                image_path = input("Enter image path: ").strip()
                
                confidence = input("Enter confidence threshold (default 70): ").strip()
                confidence_threshold = float(confidence) if confidence else 70
                
                try:
                    tester.test_image(image_path, confidence_threshold)
                except Exception as e:
                    print(f"Error testing image: {str(e)}")
                
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"Testing failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Auto mode for quick testing
        try:
            tester = FaceRecognitionTester()
            print("Starting 30-second auto test...")
            tester.test_realtime(confidence_threshold=70, max_duration=30)
        except Exception as e:
            print(f"Auto test failed: {str(e)}")
    else:
        main()