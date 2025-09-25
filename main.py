import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import csv
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import threading
import subprocess
import sys

class StudentMonitoringSystem:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AI-Based Student Monitoring System")
        self.window.geometry('1280x720')
        self.window.configure(background='gray85')
        
        # Create necessary directories
        self.create_directories()
        
        # Initialize UI
        self.setup_ui()
        
        # Variables for face recognition
        self.recognizer = None
        self.detector = None
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            'TrainingImage',
            'TrainingImageLabel',
            'StudentDetails',
            'Attendance_management/Attendance/spcc',
            'Attendance_management/Attendance/css',
            'Attendance_management/Attendance/ai',
            'Attendance_management/Attendance/mc',
            'Attendance_management/Attendance/Manually Attendance'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main title
        self.message = tk.Label(
            self.window, 
            text="AI-Based Student Monitoring", 
            bg="turquoise4", 
            fg="white", 
            width=45, 
            height=3, 
            font=('times', 30, 'bold')
        )
        self.message.place(x=80, y=20)
        
        # Notification label
        self.notification = tk.Label(
            self.window, 
            text="System Ready", 
            bg="Green", 
            fg="white", 
            width=15, 
            height=3, 
            font=('times', 17)
        )
        self.notification.place(x=20, y=400)
        
        # Input fields
        self.setup_input_fields()
        
        # Buttons
        self.setup_buttons()
    
    def setup_input_fields(self):
        """Setup input fields for enrollment and name"""
        # Enrollment input
        lbl_enrollment = tk.Label(
            self.window, 
            text="Enter Enrollment : ", 
            width=20, 
            height=2,
            fg="black", 
            bg="misty rose", 
            font=('times', 15, 'bold')
        )
        lbl_enrollment.place(x=200, y=200)
        
        # Validate enrollment input (numbers only)
        vcmd = (self.window.register(self.validate_number), '%P', '%d')
        self.txt_enrollment = tk.Entry(
            self.window, 
            validate="key", 
            validatecommand=vcmd,
            width=20, 
            bg="white",
            fg="black", 
            font=('times', 25)
        )
        self.txt_enrollment.place(x=550, y=210)
        
        # Name input
        lbl_name = tk.Label(
            self.window, 
            text="Enter Name : ", 
            width=20, 
            fg="black",
            bg="misty rose", 
            height=2, 
            font=('times', 15, 'bold')
        )
        lbl_name.place(x=200, y=300)
        
        self.txt_name = tk.Entry(
            self.window, 
            width=20, 
            bg="white",
            fg="black", 
            font=('times', 25)
        )
        self.txt_name.place(x=550, y=310)
        
        # Clear buttons
        clear_enrollment = tk.Button(
            self.window, 
            text="Clear", 
            command=self.clear_enrollment, 
            fg="white", 
            bg="cyan4",
            width=10, 
            height=1, 
            activebackground="white", 
            font=('times', 15, 'bold')
        )
        clear_enrollment.place(x=950, y=210)
        
        clear_name = tk.Button(
            self.window, 
            text="Clear", 
            command=self.clear_name, 
            fg="white", 
            bg="cyan4",
            width=10, 
            height=1, 
            activebackground="white", 
            font=('times', 15, 'bold')
        )
        clear_name.place(x=950, y=310)
    
    def setup_buttons(self):
        """Setup main action buttons"""
        # Take Images button
        btn_take_images = tk.Button(
            self.window, 
            text="Take Images", 
            command=self.take_images_thread, 
            fg="black", 
            bg="turquoise1",
            width=20, 
            height=3, 
            activebackground="white", 
            font=('times', 15, 'bold')
        )
        btn_take_images.place(x=90, y=500)
        
        # Train Model button
        btn_train = tk.Button(
            self.window, 
            text="Train Model", 
            fg="black", 
            command=self.train_model_thread, 
            bg="turquoise1",
            width=20, 
            height=3, 
            activebackground="white", 
            font=('times', 15, 'bold')
        )
        btn_train.place(x=390, y=500)
        
        # Automatic Attendance button
        btn_attendance = tk.Button(
            self.window, 
            text="Automatic Attendance", 
            fg="black", 
            command=self.automatic_attendance,
            bg="turquoise1", 
            width=20, 
            height=3, 
            activebackground="white", 
            font=('times', 15, 'bold')
        )
        btn_attendance.place(x=690, y=500)
        
        # Manual Attendance button
        btn_manual = tk.Button(
            self.window, 
            text="Manual Attendance", 
            command=self.manual_attendance, 
            fg="black",
            bg="turquoise1", 
            width=20, 
            height=3, 
            activebackground="white", 
            font=('times', 15, 'bold')
        )
        btn_manual.place(x=990, y=500)
        
        # Check Students button
        btn_check_students = tk.Button(
            self.window, 
            text="Check Registered Students", 
            command=self.check_students, 
            fg="black",
            bg="turquoise1", 
            width=19, 
            height=1, 
            activebackground="white", 
            font=('times', 15, 'bold')
        )
        btn_check_students.place(x=990, y=410)
    
    def validate_number(self, value, action):
        """Validate that input contains only numbers"""
        if action == '1':  # insert
            return value.isdigit() or value == ""
        return True
    
    def clear_enrollment(self):
        """Clear enrollment field"""
        self.txt_enrollment.delete(0, tk.END)
    
    def clear_name(self):
        """Clear name field"""
        self.txt_name.delete(0, tk.END)
    
    def update_notification(self, message, color="Green"):
        """Update notification label"""
        self.notification.configure(text=message, bg=color)
        self.window.update()
    
    def show_error(self, title, message):
        """Show error dialog"""
        messagebox.showerror(title, message)
    
    def show_info(self, title, message):
        """Show info dialog"""
        messagebox.showinfo(title, message)
    
    def take_images_thread(self):
        """Start image capture in a separate thread"""
        thread = threading.Thread(target=self.take_images)
        thread.daemon = True
        thread.start()
    
    def take_images(self):
        """Capture student images for training"""
        enrollment = self.txt_enrollment.get().strip()
        name = self.txt_name.get().strip()
        
        # Validation
        if not enrollment:
            self.show_error("Error", "Please enter enrollment number")
            return
        
        if not name:
            self.show_error("Error", "Please enter student name")
            return
        
        if len(enrollment) < 4:
            self.show_error("Error", "Enrollment number should be at least 4 digits")
            return
        
        try:
            # Initialize camera
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                self.show_error("Error", "Could not open camera")
                return
            
            # Initialize face detector
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            
            sample_num = 0
            max_samples = 100
            
            self.update_notification("Taking images... Press 'q' to stop", "orange")
            
            while sample_num < max_samples:
                ret, img = cam.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sample_num += 1
                    
                    # Save the captured face
                    filename = f"TrainingImage/{name}.{enrollment}.{sample_num}.jpg"
                    cv2.imwrite(filename, gray[y:y+h, x:x+w])
                    
                    # Show progress
                    cv2.putText(img, f"Samples: {sample_num}/{max_samples}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Capturing Images - Press q to quit', img)
                
                if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= max_samples:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            # Save student details to CSV
            self.save_student_details(enrollment, name)
            
            self.update_notification(f"Images saved for {name} ({enrollment})", "SpringGreen3")
            self.show_info("Success", f"Successfully captured {sample_num} images for {name}")
            
        except Exception as e:
            self.show_error("Error", f"An error occurred: {str(e)}")
            self.update_notification("Error capturing images", "Red")
    
    def save_student_details(self, enrollment, name):
        """Save student details to CSV file"""
        try:
            timestamp = datetime.datetime.now()
            date = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H:%M:%S')
            
            csv_file = "StudentDetails/StudentDetails.csv"
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, "a+", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Enrollment", "Name", "Date", "Time"])
                writer.writerow([enrollment, name, date, time_str])
                
        except Exception as e:
            print(f"Error saving student details: {e}")
    
    def train_model_thread(self):
        """Start model training in a separate thread"""
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    
    def train_model(self):
        """Train the face recognition model"""
        try:
            self.update_notification("Training model... Please wait", "orange")
            
            # Check if training images exist
            if not os.path.exists('TrainingImage') or not os.listdir('TrainingImage'):
                self.show_error("Error", "No training images found. Please capture images first.")
                self.update_notification("No training data found", "Red")
                return
            
            # Initialize recognizer and detector
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            
            # Get images and labels
            faces, ids = self.get_images_and_labels("TrainingImage", detector)
            
            if len(faces) == 0:
                self.show_error("Error", "No faces found in training images")
                self.update_notification("Training failed", "Red")
                return
            
            # Train the recognizer
            recognizer.train(faces, np.array(ids))
            
            # Save the trained model
            model_path = "TrainingImageLabel/trainer.yml"
            recognizer.save(model_path)
            
            self.update_notification("Model trained successfully", "SpringGreen3")
            self.show_info("Success", f"Model trained with {len(faces)} face samples")
            
        except Exception as e:
            self.show_error("Error", f"Training failed: {str(e)}")
            self.update_notification("Training failed", "Red")
    
    def get_images_and_labels(self, path, detector):
        """Get images and corresponding labels from the training directory"""
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        face_samples = []
        ids = []
        
        for image_path in image_paths:
            try:
                # Load and convert image
                pil_image = Image.open(image_path).convert('L')
                image_np = np.array(pil_image, 'uint8')
                
                # Extract ID from filename
                filename = os.path.split(image_path)[-1]
                id_str = filename.split(".")[1]
                student_id = int(id_str)
                
                # Detect faces in the image
                faces = detector.detectMultiScale(image_np, 1.2, 5)
                
                # Add face samples and IDs
                for (x, y, w, h) in faces:
                    face_samples.append(image_np[y:y+h, x:x+w])
                    ids.append(student_id)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return face_samples, ids
    
    def automatic_attendance(self):
        """Start automatic attendance system"""
        subject_window = SubjectWindow(self)
    
    def manual_attendance(self):
        """Open manual attendance file"""
        try:
            manual_file = "Attendance_management/Attendance/Manually Attendance/Manual_Attendance.xlsx"
            if os.path.exists(manual_file):
                os.startfile(manual_file)
            else:
                # Create a basic Excel file if it doesn't exist
                import pandas as pd
                df = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Subject', 'Status'])
                df.to_excel(manual_file, index=False)
                os.startfile(manual_file)
        except Exception as e:
            self.show_error("Error", f"Could not open manual attendance file: {str(e)}")
    
    def check_students(self):
        """Display registered students"""
        AdminWindow(self)
    
    def run(self):
        """Start the application"""
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.window.destroy()


class SubjectWindow:
    """Window for subject selection and attendance marking"""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel()
        self.window.title("Subject Selection")
        self.window.geometry('600x400')
        self.window.configure(background='grey80')
        self.window.grab_set()  # Make window modal
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup subject selection UI"""
        # Title
        title = tk.Label(
            self.window, 
            text="Select Subject and Mark Attendance",
            bg="grey80", 
            fg="black", 
            font=('times', 18, 'bold')
        )
        title.pack(pady=20)
        
        # Subject input
        subject_frame = tk.Frame(self.window, bg='grey80')
        subject_frame.pack(pady=10)
        
        tk.Label(
            subject_frame, 
            text="Enter Subject:", 
            bg="grey80", 
            fg="black",
            font=('times', 15, 'bold')
        ).pack(side=tk.LEFT, padx=10)
        
        self.subject_entry = tk.Entry(
            subject_frame, 
            width=20, 
            bg="white",
            fg="black", 
            font=('times', 15)
        )
        self.subject_entry.pack(side=tk.LEFT, padx=10)
        
        # Duration input
        duration_frame = tk.Frame(self.window, bg='grey80')
        duration_frame.pack(pady=10)
        
        tk.Label(
            duration_frame, 
            text="Duration (seconds):", 
            bg="grey80", 
            fg="black",
            font=('times', 15, 'bold')
        ).pack(side=tk.LEFT, padx=10)
        
        self.duration_entry = tk.Entry(
            duration_frame, 
            width=10, 
            bg="white",
            fg="black", 
            font=('times', 15)
        )
        self.duration_entry.insert(0, "30")  # Default 30 seconds
        self.duration_entry.pack(side=tk.LEFT, padx=10)
        
        # Buttons
        button_frame = tk.Frame(self.window, bg='grey80')
        button_frame.pack(pady=30)
        
        start_btn = tk.Button(
            button_frame,
            text="Start Attendance", 
            command=self.start_attendance,
            fg="white", 
            bg="green", 
            width=15, 
            height=2,
            font=('times', 15, 'bold')
        )
        start_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel", 
            command=self.window.destroy,
            fg="white", 
            bg="red", 
            width=15, 
            height=2,
            font=('times', 15, 'bold')
        )
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # Notification
        self.notification = tk.Label(
            self.window,
            text="Enter subject name and duration", 
            bg="grey80", 
            fg="black",
            font=('times', 12)
        )
        self.notification.pack(pady=10)
    
    def start_attendance(self):
        """Start the face recognition attendance system"""
        subject = self.subject_entry.get().strip()
        
        if not subject:
            messagebox.showerror("Error", "Please enter subject name")
            return
        
        try:
            duration = int(self.duration_entry.get())
            if duration <= 0:
                raise ValueError("Duration must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid duration in seconds")
            return
        
        self.window.destroy()
        
        # Start attendance marking in a separate thread
        thread = threading.Thread(target=self.mark_attendance, args=(subject, duration))
        thread.daemon = True
        thread.start()
    
    def mark_attendance(self, subject, duration):
        """Mark attendance using face recognition"""
        try:
            # Check if model exists
            model_path = "TrainingImageLabel/trainer.yml"
            if not os.path.exists(model_path):
                messagebox.showerror("Error", "Trained model not found. Please train the model first.")
                return
            
            # Load the trained model
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(model_path)
            
            # Load face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            
            # Load student details
            student_details = self.load_student_details()
            if not student_details:
                messagebox.showerror("Error", "No student details found")
                return
            
            # Initialize camera
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            # Attendance tracking
            attendance = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])
            start_time = time.time()
            
            self.parent.update_notification("Marking attendance... Press ESC to stop", "orange")
            
            while time.time() - start_time < duration:
                ret, frame = cam.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                
                for (x, y, w, h) in faces:
                    student_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < 70:  # Recognized face
                        name = student_details.get(student_id, f"Unknown_{student_id}")
                        
                        # Add to attendance if not already present
                        if not ((attendance['Enrollment'] == student_id).any()):
                            timestamp = datetime.datetime.now()
                            date_str = timestamp.strftime('%Y-%m-%d')
                            time_str = timestamp.strftime('%H:%M:%S')
                            
                            new_row = pd.DataFrame({
                                'Enrollment': [student_id],
                                'Name': [name],
                                'Date': [date_str],
                                'Time': [time_str]
                            })
                            attendance = pd.concat([attendance, new_row], ignore_index=True)
                        
                        # Draw rectangle and name
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({student_id})", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        # Unknown face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Show remaining time
                remaining = int(duration - (time.time() - start_time))
                cv2.putText(frame, f"Time remaining: {remaining}s", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Attendance count: {len(attendance)}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                cv2.imshow('Attendance - Press ESC to stop', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            # Save attendance
            self.save_attendance(attendance, subject)
            
            self.parent.update_notification("Attendance marked successfully", "SpringGreen3")
            
            # Show attendance summary
            self.show_attendance_summary(attendance, subject)
            
        except Exception as e:
            messagebox.showerror("Error", f"Attendance marking failed: {str(e)}")
            self.parent.update_notification("Attendance marking failed", "Red")
    
    def load_student_details(self):
        """Load student details from CSV"""
        try:
            csv_file = "StudentDetails/StudentDetails.csv"
            if not os.path.exists(csv_file):
                return {}
            
            df = pd.read_csv(csv_file)
            return dict(zip(df['Enrollment'], df['Name']))
        except Exception as e:
            print(f"Error loading student details: {e}")
            return {}
    
    def save_attendance(self, attendance_df, subject):
        """Save attendance to CSV file"""
        try:
            if attendance_df.empty:
                messagebox.showwarning("Warning", "No attendance recorded")
                return
            
            timestamp = datetime.datetime.now()
            date_str = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H-%M-%S')
            
            # Create subject directory
            subject_dir = f"Attendance_management/Attendance/{subject.lower()}"
            os.makedirs(subject_dir, exist_ok=True)
            
            # Save to CSV
            filename = f"{subject_dir}/{subject}_{date_str}_{time_str}.csv"
            attendance_df.to_csv(filename, index=False)
            
            print(f"Attendance saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving attendance: {e}")
    
    def show_attendance_summary(self, attendance_df, subject):
        """Show attendance summary in a new window"""
        summary_window = tk.Toplevel()
        summary_window.title(f"Attendance Summary - {subject}")
        summary_window.geometry('600x400')
        summary_window.configure(background='white')
        
        # Title
        title = tk.Label(
            summary_window,
            text=f"Attendance Summary for {subject}",
            bg="white",
            fg="black",
            font=('times', 16, 'bold')
        )
        title.pack(pady=10)
        
        # Create treeview for attendance data
        from tkinter import ttk
        
        frame = tk.Frame(summary_window)
        frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        tree = ttk.Treeview(frame, columns=('Enrollment', 'Name', 'Date', 'Time'), show='headings')
        tree.heading('Enrollment', text='Enrollment')
        tree.heading('Name', text='Name')
        tree.heading('Date', text='Date')
        tree.heading('Time', text='Time')
        
        # Add data to treeview
        for _, row in attendance_df.iterrows():
            tree.insert('', 'end', values=(row['Enrollment'], row['Name'], row['Date'], row['Time']))
        
        tree.pack(side='left', fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Summary info
        summary_info = tk.Label(
            summary_window,
            text=f"Total Present: {len(attendance_df)}",
            bg="white",
            fg="black",
            font=('times', 12, 'bold')
        )
        summary_info.pack(pady=10)


class AdminWindow:
    """Window for viewing registered students"""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel()
        self.window.title("Registered Students")
        self.window.geometry('800x600')
        self.window.configure(background='white')
        self.window.grab_set()
        
        self.setup_ui()
        self.load_students()
    
    def setup_ui(self):
        """Setup admin UI"""
        # Title
        title = tk.Label(
            self.window,
            text="Registered Students",
            bg="white",
            fg="black",
            font=('times', 18, 'bold')
        )
        title.pack(pady=10)
        
        # Create treeview
        from tkinter import ttk
        
        frame = tk.Frame(self.window)
        frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        self.tree = ttk.Treeview(frame, columns=('Enrollment', 'Name', 'Date', 'Time'), show='headings')
        self.tree.heading('Enrollment', text='Enrollment')
        self.tree.heading('Name', text='Name')
        self.tree.heading('Date', text='Registration Date')
        self.tree.heading('Time', text='Registration Time')
        
        self.tree.pack(side='left', fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar.set)
    
    def load_students(self):
        """Load and display student data"""
        try:
            csv_file = "StudentDetails/StudentDetails.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    self.tree.insert('', 'end', values=(row['Enrollment'], row['Name'], row['Date'], row['Time']))
            else:
                messagebox.showinfo("Info", "No student records found")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load student data: {str(e)}")


def main():
    """Main function to start the application"""
    try:
        app = StudentMonitoringSystem()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")


if __name__ == '__main__':
    main()