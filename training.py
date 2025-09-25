import cv2
import os
import numpy as np
from PIL import Image
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceTrainer:
    """Class for training face recognition models"""
    
    def __init__(self, training_path="TrainingImage", model_path="TrainingImageLabel"):
        self.training_path = Path(training_path)
        self.model_path = Path(model_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Create directories if they don't exist
        self.model_path.mkdir(exist_ok=True)
        
    def validate_training_data(self):
        """Validate that training data exists and is properly formatted"""
        if not self.training_path.exists():
            raise FileNotFoundError(f"Training image directory '{self.training_path}' not found")
        
        image_files = list(self.training_path.glob("*.jpg")) + list(self.training_path.glob("*.jpeg")) + list(self.training_path.glob("*.png"))
        
        if not image_files:
            raise ValueError(f"No image files found in '{self.training_path}'")
        
        # Check filename format
        valid_files = []
        invalid_files = []
        
        for img_file in image_files:
            parts = img_file.stem.split('.')
            if len(parts) >= 3:  # name.id.number format
                try:
                    int(parts[1])  # Check if ID is numeric
                    valid_files.append(img_file)
                except ValueError:
                    invalid_files.append(img_file)
            else:
                invalid_files.append(img_file)
        
        if invalid_files:
            logger.warning(f"Found {len(invalid_files)} files with invalid naming format:")
            for file in invalid_files[:5]:  # Show first 5 invalid files
                logger.warning(f"  - {file.name}")
            if len(invalid_files) > 5:
                logger.warning(f"  ... and {len(invalid_files) - 5} more")
        
        if not valid_files:
            raise ValueError("No valid training images found. Images should be named as 'name.id.number.jpg'")
        
        logger.info(f"Found {len(valid_files)} valid training images")
        return valid_files
    
    def get_images_and_labels(self, image_files):
        """Extract face samples and corresponding labels from images"""
        face_samples = []
        ids = []
        processed_count = 0
        error_count = 0
        
        logger.info("Processing training images...")
        
        for image_path in image_files:
            try:
                # Load and convert image to grayscale
                pil_image = Image.open(image_path).convert('L')
                image_np = np.array(pil_image, 'uint8')
                
                # Extract ID from filename (name.id.number.ext format)
                filename = image_path.stem
                parts = filename.split('.')
                student_id = int(parts[1])
                
                # Detect faces in the image
                faces = self.detector.detectMultiScale(
                    image_np, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                # Process each detected face
                face_found = False
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = image_np[y:y+h, x:x+w]
                    
                    # Resize face to standard size for better recognition
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    face_samples.append(face_roi)
                    ids.append(student_id)
                    face_found = True
                
                if face_found:
                    processed_count += 1
                else:
                    logger.warning(f"No face detected in {image_path.name}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {str(e)}")
                error_count += 1
                continue
        
        logger.info(f"Successfully processed {processed_count} images")
        if error_count > 0:
            logger.warning(f"Failed to process {error_count} images")
        
        if not face_samples:
            raise ValueError("No face samples could be extracted from the training images")
        
        return face_samples, ids
    
    def train_model(self, face_samples, ids):
        """Train the LBPH face recognizer"""
        logger.info(f"Training model with {len(face_samples)} face samples...")
        
        # Convert to numpy array
        ids = np.array(ids, dtype=np.int32)
        
        # Train the recognizer
        self.recognizer.train(face_samples, ids)
        
        # Save the trained model
        model_file = self.model_path / "trainer.yml"
        self.recognizer.save(str(model_file))
        
        logger.info(f"Model saved to {model_file}")
        
        # Print training statistics
        unique_ids = np.unique(ids)
        logger.info(f"Training completed:")
        logger.info(f"  - Total samples: {len(face_samples)}")
        logger.info(f"  - Unique students: {len(unique_ids)}")
        logger.info(f"  - Average samples per student: {len(face_samples) / len(unique_ids):.1f}")
        
        return model_file
    
    def train(self):
        """Complete training pipeline"""
        try:
            logger.info("Starting face recognition model training...")
            
            # Validate training data
            valid_files = self.validate_training_data()
            
            # Extract face samples and labels
            face_samples, ids = self.get_images_and_labels(valid_files)
            
            # Train the model
            model_file = self.train_model(face_samples, ids)
            
            logger.info("Training completed successfully!")
            return model_file
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """Main function for training"""
    try:
        # Initialize trainer
        trainer = FaceTrainer()
        
        # Train the model
        model_file = trainer.train()
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model saved to: {model_file}")
        print("You can now use the model for face recognition.")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)

def test_model(model_path="TrainingImageLabel/trainer.yml"):
    """Test the trained model with a simple validation"""
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        # Load the model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
        
        print(f"Model loaded successfully from {model_path}")
        
        # Try to get some basic info about the model
        # Note: LBPH doesn't provide direct access to training data info
        print("Model validation completed")
        return True
        
    except Exception as e:
        print(f"Model validation failed: {str(e)}")
        return False

if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_model()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python training.py          # Train the model")
            print("  python training.py --test   # Test the trained model")
            print("  python training.py --help   # Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        main()