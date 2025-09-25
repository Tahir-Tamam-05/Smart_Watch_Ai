import pymysql
import os
import glob
import csv
import smtplib
import base64
from io import BytesIO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
import hashlib
import secrets
from contextlib import contextmanager

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Shared storage for cross-process consistency
from storage import read_students as storage_read_students, upsert_student as storage_upsert_student, delete_student as storage_delete_student

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Generate a secure secret key

# Configuration
class Config:
    DB_HOST = 'localhost'
    DB_USER = 'root'
    DB_PASSWORD = ''
    DB_NAME = 'ai_based_student_monitoring'
    DB_PORT = 3307
    
    # Email configuration
    EMAIL_HOST = 'smtp.gmail.com'
    EMAIL_PORT = 587
    EMAIL_USER = 'your_email@gmail.com'  # Replace with your email
    EMAIL_PASSWORD = 'your_app_password'  # Replace with your app password
    
    # Admin credentials (in production, use environment variables)
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD_HASH = hashlib.sha256('admin123'.encode()).hexdigest()

config = Config()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = pymysql.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
            port=config.DB_PORT,
            charset='utf8mb4'
        )
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        app.logger.error(f"Database error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def validate_email(email):
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Basic phone validation"""
    import re
    pattern = r'^[0-9]{10,15}$'
    return re.match(pattern, phone.replace('+', '').replace('-', '').replace(' ', '')) is not None

# ---------- CSV student storage helpers (delegate to storage.py) ----------
read_students_csv = storage_read_students

def upsert_student_csv(record):
    return storage_upsert_student(record)

def delete_student_csv(moodle_id):
    return storage_delete_student(moodle_id)

############################## HOME PAGE ################################
@app.route('/', methods=['GET'])
def home():
    if 'admin_logged_in' in session:
        return render_template('home.html', button='logout')
    else:
        return render_template('home.html', button='admin_login')

############################## STUDENT FORM PAGE ################################
@app.route('/studentform', methods=['GET', 'POST'])
def studentform():
    if request.method == 'POST':
        try:
            # Get and validate form data
            name = request.form.get('name', '').strip()
            moodle_id = request.form.get('moodle_id', '').strip()
            email = request.form.get('email', '').strip()
            parent_email = request.form.get('parent_email', '').strip()
            contact_number = request.form.get('contact_number', '').strip()

            # Validation
            errors = []
            if not name or len(name) < 2:
                errors.append("Name must be at least 2 characters long")
            if not moodle_id or not moodle_id.isdigit():
                errors.append("Moodle ID must be a valid number")
            if not validate_email(email):
                errors.append("Invalid email format")
            if not validate_email(parent_email):
                errors.append("Invalid parent email format")
            if not validate_phone(contact_number):
                errors.append("Invalid contact number format")

            if errors:
                for error in errors:
                    flash(error, 'error')
                return render_template('studentform.html')

            # Insert data into database (fallback to CSV if DB unavailable)
            db_inserted = False
            try:
                with get_db_connection() as conn:
                    cur = conn.cursor()
                    # Check if moodle_id already exists
                    cur.execute('SELECT COUNT(*) FROM student_info WHERE moodle_id = %s', (moodle_id,))
                    if cur.fetchone()[0] > 0:
                        flash('Moodle ID already exists!', 'error')
                        return render_template('studentform.html')
                    
                    cur.execute('''INSERT INTO student_info 
                                  (name, moodle_id, email, parent_email, contact_number) 
                                  VALUES (%s, %s, %s, %s, %s)''',
                               (name, moodle_id, email, parent_email, contact_number))
                    conn.commit()
                    db_inserted = True
            except Exception as e:
                app.logger.warning(f"DB unavailable for studentform insert, using CSV: {e}")
                # Check duplicate in CSV
                existing = [row for row in read_students_csv() if str(row[1]).strip() == str(moodle_id).strip()]
                if existing:
                    flash('Moodle ID already exists!', 'error')
                    return render_template('studentform.html')

            # Append data to CSV file
            csv_data = {
                'Name': name, 
                'Moodle ID': moodle_id, 
                'Email ID': email,
                "Parent's Email": parent_email, 
                'Contact Number': contact_number
            }
            
            file_path = 'studentdetails.csv'
            file_exists = os.path.isfile(file_path)
            
            with open(file_path, 'a', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['Name', 'Moodle ID', 'Email ID', "Parent's Email", 'Contact Number']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_data)

            flash('Form submitted successfully!', 'success')
            return render_template('studentform.html')
            
        except Exception as e:
            app.logger.error(f"Error in studentform: {str(e)}")
            flash('An error occurred while processing your request', 'error')
            return render_template('studentform.html')
    
    return render_template('studentform.html')

############################## ADMIN LOGIN PAGE ################################
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('admin_login.html')

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if username == config.ADMIN_USERNAME and password_hash == config.ADMIN_PASSWORD_HASH:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('admin_login.html')

############################## ADMIN DASHBOARD ################################
@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM student_info ORDER BY name')
            data = cursor.fetchall()
        
        return render_template('admin_dashboard.html', students=data)
    except Exception as e:
        app.logger.error(f"Error in admin_dashboard: {str(e)}")
        csv_students = read_students_csv()
        if csv_students:
            flash('Loaded student data from file (DB unavailable)', 'warning')
            return render_template('admin_dashboard.html', students=csv_students)
        flash('Error loading student data', 'error')
        return render_template('admin_dashboard.html', students=[])

@app.route('/add_student', methods=['POST'])
def add_student():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    try:
        name = request.form.get('name', '').strip()
        moodle_id = request.form.get('moodle_id', '').strip()
        email = request.form.get('email', '').strip()
        parent_email = request.form.get('parent_email', '').strip()
        contact_number = request.form.get('contact_number', '').strip()

        # Validation
        if not all([name, moodle_id, email, parent_email, contact_number]):
            flash('All fields are required', 'error')
            return redirect(url_for('admin_dashboard'))

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO student_info 
                             (name, moodle_id, email, parent_email, contact_number) 
                             VALUES (%s, %s, %s, %s, %s)''',
                          (name, moodle_id, email, parent_email, contact_number))
            conn.commit()
        
        flash('Student added successfully', 'success')
    except Exception as e:
        app.logger.error(f"Error adding student: {str(e)}")
        try:
            upsert_student_csv({
                'Name': name,
                'Moodle ID': moodle_id,
                'Email ID': email,
                "Parent's Email": parent_email,
                'Contact Number': contact_number
            })
            flash('Student added to file (DB unavailable)', 'warning')
        except Exception as e2:
            app.logger.error(f"CSV add error: {e2}")
            flash('Error adding student', 'error')

    return redirect(url_for('admin_dashboard'))

@app.route('/update_student/<id>', methods=['POST'])
def update_student(id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    try:
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        parent_email = request.form.get('parent_email', '').strip()
        contact_number = request.form.get('contact_number', '').strip()

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''UPDATE student_info 
                             SET name=%s, email=%s, parent_email=%s, contact_number=%s 
                             WHERE moodle_id=%s''',
                          (name, email, parent_email, contact_number, id))
            conn.commit()
        
        flash('Student updated successfully', 'success')
    except Exception as e:
        app.logger.error(f"Error updating student: {str(e)}")
        try:
            upsert_student_csv({
                'Name': name,
                'Moodle ID': id,
                'Email ID': email,
                "Parent's Email": parent_email,
                'Contact Number': contact_number
            })
            flash('Student updated in file (DB unavailable)', 'warning')
        except Exception as e2:
            app.logger.error(f"CSV update error: {e2}")
            flash('Error updating student', 'error')

    return redirect(url_for('admin_dashboard'))

@app.route('/delete_student/<id>', methods=['POST'])
def delete_student(id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM student_info WHERE moodle_id=%s', (id,))
            conn.commit()
        
        flash('Student deleted successfully', 'success')
    except Exception as e:
        app.logger.error(f"Error deleting student: {str(e)}")
        try:
            delete_student_csv(id)
            flash('Student deleted from file (DB unavailable)', 'warning')
        except Exception as e2:
            app.logger.error(f"CSV delete error: {e2}")
            flash('Error deleting student', 'error')

    return redirect(url_for('admin_dashboard'))

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

############################## ATTENDANCE PROCESSING ################################
def process_attendance_data(subject_dir, total_classes):
    """Process attendance data for a specific subject"""
    try:
        # Validate inputs
        total_classes = int(total_classes)
        if total_classes <= 0:
            raise ValueError("total_classes must be > 0")

        # Paths
        student_details_dir = "StudentDetails"
        student_details_file = "studentdetails.csv"

        # Validate subject directory
        if not os.path.isdir(subject_dir):
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

        # Read subject attendance CSVs
        subject_files = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.lower().endswith('.csv')]
        if not subject_files:
            raise FileNotFoundError("No attendance files found for this subject")

        subject_dfs = []
        for csv_file in subject_files:
            try:
                df = pd.read_csv(csv_file)
                subject_dfs.append(df)
            except Exception as e:
                app.logger.error(f"Failed reading subject CSV {csv_file}: {e}")
        if not subject_dfs:
            raise ValueError("No readable attendance data in subject files")

        subject_df = pd.concat(subject_dfs, ignore_index=True)

        # Ensure a consistent Enrollment column
        possible_enrollment_cols = [c for c in subject_df.columns if c.strip().lower() in ("enrollment", "moodle id", "moodle_id", "roll", "rollno", "roll_no")]
        if not possible_enrollment_cols:
            raise KeyError("No Enrollment-like column found in subject CSVs")
        enrollment_col = possible_enrollment_cols[0]
        subject_df = subject_df.rename(columns={enrollment_col: "Enrollment"})

        # Compute counts per Enrollment
        counts = subject_df.dropna(subset=["Enrollment"]).groupby("Enrollment").size().reset_index(name="Count")
        # Preserve original subtract-1 logic but not below zero
        counts["Count"] = (counts["Count"] - 1).clip(lower=0)

        # Build student names mapping from available sources
        names_sources = []

        # Root student details file
        if os.path.exists(student_details_file):
            try:
                df = pd.read_csv(student_details_file)
                names_sources.append(df)
            except Exception as e:
                app.logger.error(f"Failed reading {student_details_file}: {e}")

        # Any CSVs under StudentDetails directory
        if os.path.isdir(student_details_dir):
            student_files = [os.path.join(student_details_dir, f) for f in os.listdir(student_details_dir) if f.lower().endswith('.csv')]
            for csv_file in student_files:
                try:
                    df = pd.read_csv(csv_file)
                    names_sources.append(df)
                except Exception as e:
                    app.logger.error(f"Failed reading student CSV {csv_file}: {e}")

        if names_sources:
            names_df = pd.concat(names_sources, ignore_index=True)
            # Identify ID column and normalize to Enrollment
            id_col = None
            for candidate in ["Enrollment", "Moodle ID", "moodle_id", "ID", "Id"]:
                if candidate in names_df.columns:
                    id_col = candidate
                    break
            if id_col is not None:
                names_df = names_df.rename(columns={id_col: "Enrollment"})
                if "Name" not in names_df.columns:
                    names_df["Name"] = None
                names_df = names_df[["Enrollment", "Name"]].dropna(subset=["Enrollment"]).drop_duplicates(subset=["Enrollment"], keep="last")
            else:
                names_df = pd.DataFrame(columns=["Enrollment", "Name"])
        else:
            names_df = pd.DataFrame(columns=["Enrollment", "Name"])

        # Merge counts with names using string keys to avoid dtype mismatches
        counts["Enrollment"] = counts["Enrollment"].astype(str).str.strip()
        names_df["Enrollment"] = names_df["Enrollment"].astype(str).str.strip()

        result_df = counts.merge(names_df, on="Enrollment", how="left")

        # Add totals and percentage
        result_df["Total Classes"] = total_classes
        result_df["Percentage"] = (result_df["Count"] / result_df["Total Classes"] * 100).round(2)

        # Reorder columns
        result_df = result_df[["Enrollment", "Name", "Count", "Total Classes", "Percentage"]]

        return result_df

    except Exception as e:
        app.logger.error(f"Error processing attendance: {str(e)}")
        raise

@app.route('/studentattendance', methods=['GET'])
def studentattendance():
    return render_template('student_attendance.html')

# Subject-specific routes with improved error handling
@app.route('/spcc', methods=['POST'])
def spcc():
    try:
        total_classes = int(request.form.get('total_classes', 0))
        if total_classes <= 0:
            flash('Please enter a valid number of total classes', 'error')
            return redirect('/studentattendance')

        subject_dir = "Attendance_management/Attendance/spcc/"
        result_df = process_attendance_data(subject_dir, total_classes)
        result_df.to_csv("attendance_summary_spcc.csv", index=False)
        
        return redirect('/table')
    except Exception as e:
        app.logger.error(f"Error in SPCC attendance: {str(e)}")
        flash('Error processing SPCC attendance data', 'error')
        return redirect('/studentattendance')

@app.route('/css', methods=['POST'])
def css():
    try:
        total_classes = int(request.form.get('total_classes', 0))
        if total_classes <= 0:
            flash('Please enter a valid number of total classes', 'error')
            return redirect('/studentattendance')

        subject_dir = "Attendance_management/Attendance/css/"
        result_df = process_attendance_data(subject_dir, total_classes)
        result_df.to_csv("attendance_summary_css.csv", index=False)
        
        return redirect('/table1')
    except Exception as e:
        app.logger.error(f"Error in CSS attendance: {str(e)}")
        flash('Error processing CSS attendance data', 'error')
        return redirect('/studentattendance')

@app.route('/ai', methods=['POST'])
def ai():
    try:
        total_classes = int(request.form.get('total_classes', 0))
        if total_classes <= 0:
            flash('Please enter a valid number of total classes', 'error')
            return redirect('/studentattendance')

        subject_dir = "Attendance_management/Attendance/ai/"
        result_df = process_attendance_data(subject_dir, total_classes)
        result_df.to_csv("attendance_summary_ai.csv", index=False)
        
        return redirect('/table2')
    except Exception as e:
        app.logger.error(f"Error in AI attendance: {str(e)}")
        flash('Error processing AI attendance data', 'error')
        return redirect('/studentattendance')

@app.route('/mc', methods=['POST'])
def mc():
    try:
        total_classes = int(request.form.get('total_classes', 0))
        if total_classes <= 0:
            flash('Please enter a valid number of total classes', 'error')
            return redirect('/studentattendance')

        subject_dir = "Attendance_management/Attendance/mc/"
        result_df = process_attendance_data(subject_dir, total_classes)
        result_df.to_csv("attendance_summary_mc.csv", index=False)
        
        return redirect('/table3')
    except Exception as e:
        app.logger.error(f"Error in MC attendance: {str(e)}")
        flash('Error processing MC attendance data', 'error')
        return redirect('/studentattendance')

# Table display routes
@app.route('/table')
def display_table():
    try:
        df = pd.read_csv('attendance_summary_spcc.csv')
        return render_template('table.html', data=df.to_html(index=False, classes='table table-striped'))
    except Exception as e:
        app.logger.error(f"Error displaying SPCC table: {str(e)}")
        flash('Error loading attendance data', 'error')
        return render_template('table.html', data='<p>Error loading data</p>')

@app.route('/table1')
def display_table1():
    try:
        df = pd.read_csv('attendance_summary_css.csv')
        return render_template('table.html', data=df.to_html(index=False, classes='table table-striped'))
    except Exception as e:
        app.logger.error(f"Error displaying CSS table: {str(e)}")
        flash('Error loading attendance data', 'error')
        return render_template('table.html', data='<p>Error loading data</p>')

@app.route('/table2')
def display_table2():
    try:
        df = pd.read_csv('attendance_summary_ai.csv')
        return render_template('table.html', data=df.to_html(index=False, classes='table table-striped'))
    except Exception as e:
        app.logger.error(f"Error displaying AI table: {str(e)}")
        flash('Error loading attendance data', 'error')
        return render_template('table.html', data='<p>Error loading data</p>')

@app.route('/table3')
def display_table3():
    try:
        df = pd.read_csv('attendance_summary_mc.csv')
        return render_template('table.html', data=df.to_html(index=False, classes='table table-striped'))
    except Exception as e:
        app.logger.error(f"Error displaying MC table: {str(e)}")
        flash('Error loading attendance data', 'error')
        return render_template('table.html', data='<p>Error loading data</p>')

############################## OVERALL ATTENDANCE ################################
@app.route('/overall_attendance')
def overall_attendance():
    try:
        merged_df = pd.DataFrame()
        
        attendance_files = glob.glob("attendance_summary*.csv")
        if not attendance_files:
            flash('No attendance data found. Please process subject attendance first.', 'error')
            return render_template('table.html', data='<p>No data available</p>')

        for file in attendance_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                merged_df = pd.concat([merged_df, df], ignore_index=True)

        if merged_df.empty:
            flash('No attendance data found', 'error')
            return render_template('table.html', data='<p>No data available</p>')

        overall_df = merged_df.groupby(["Enrollment", "Name"]).sum().reset_index()
        overall_df["Percentage"] = (overall_df["Count"] / overall_df["Total Classes"] * 100).round(2)
        
        overall_df.to_csv("overall_attendance.csv", index=False)
        return redirect('/table4')
        
    except Exception as e:
        app.logger.error(f"Error in overall_attendance: {str(e)}")
        flash('Error calculating overall attendance', 'error')
        return render_template('table.html', data='<p>Error calculating data</p>')

@app.route('/table4')
def display_table4():
    try:
        df = pd.read_csv('overall_attendance.csv')
        return render_template('table.html', data=df.to_html(index=False, classes='table table-striped'))
    except Exception as e:
        app.logger.error(f"Error displaying overall table: {str(e)}")
        flash('Error loading overall attendance data', 'error')
        return render_template('table.html', data='<p>Error loading data</p>')

############################## VISUALIZE ATTENDANCE ################################
@app.route('/visualize')
def visualize():
    try:
        if not os.path.exists('overall_attendance.csv'):
            flash('No overall attendance data found. Please generate it first.', 'error')
            return render_template('visualize.html', plots={})

        data = pd.read_csv('overall_attendance.csv')
        
        if data.empty:
            flash('No attendance data to visualize', 'error')
            return render_template('visualize.html', plots={})

        # Define filter conditions
        filters = {
            'above_90': {'title': 'Attendance Above 90%', 'condition': data['Percentage'] > 90},
            'below_50': {'title': 'Attendance Below 50%', 'condition': data['Percentage'] <= 50},
            'below_30': {'title': 'Attendance Below 30%', 'condition': data['Percentage'] < 30},
            'mid_range': {'title': 'Attendance Between 60% and 80%', 
                         'condition': (data['Percentage'] >= 60) & (data['Percentage'] <= 80)}
        }

        plots = {}
        for key, value in filters.items():
            try:
                filtered_data = data[value['condition']]
                
                if filtered_data.empty:
                    continue

                plt.figure(figsize=(12, 6))
                bars = plt.bar(range(len(filtered_data)), filtered_data['Percentage'], color='skyblue')
                plt.xlabel('Students')
                plt.ylabel('Attendance Percentage')
                plt.title(value['title'])
                plt.xticks(range(len(filtered_data)), filtered_data['Name'], rotation=45, ha='right')
                plt.tight_layout()

                # Add percentage labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom')

                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()

                graph = base64.b64encode(image_png).decode('utf-8')
                plots[key] = graph
                plt.close()

            except Exception as e:
                app.logger.error(f"Error creating plot {key}: {str(e)}")
                continue

        return render_template('visualize.html', plots=plots)

    except Exception as e:
        app.logger.error(f"Error in visualize: {str(e)}")
        flash('Error creating visualizations', 'error')
        return render_template('visualize.html', plots={})

############################## EMAIL FUNCTIONALITY ################################
def send_email_to_student(to_email, name):
    """Send email to student with low attendance"""
    try:
        msg = MIMEMultipart()
        msg['From'] = config.EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = 'Low Attendance Alert'
        
        message = f"""Dear {name},

Your attendance percentage is below 75%.

Please make sure to attend all classes regularly. If you have any concerns or issues affecting your attendance, please contact your academic advisor.

Thanks & Regards,
Academic Administration
APSIT"""

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(config.EMAIL_HOST, config.EMAIL_PORT)
        server.starttls()
        server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
        server.sendmail(config.EMAIL_USER, to_email, msg.as_string())
        server.quit()
        
        return True
    except Exception as e:
        app.logger.error(f"Error sending email to student {to_email}: {str(e)}")
        return False

def send_email_to_parent(to_parent_email, student_name):
    """Send email to parent about student's low attendance"""
    try:
        msg = MIMEMultipart()
        msg['From'] = config.EMAIL_USER
        msg['To'] = to_parent_email
        msg['Subject'] = 'Student Attendance Alert'
        
        message = f"""Dear Parent/Guardian,

This is to inform you that your ward {student_name}'s attendance percentage is below 75%.

Please ensure that your child attends all classes regularly. If there are any issues affecting attendance, please contact the college administration.

Thanks & Regards,
Academic Administration
APSIT"""

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(config.EMAIL_HOST, config.EMAIL_PORT)
        server.starttls()
        server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
        server.sendmail(config.EMAIL_USER, to_parent_email, msg.as_string())
        server.quit()
        
        return True
    except Exception as e:
        app.logger.error(f"Error sending email to parent {to_parent_email}: {str(e)}")
        return False

@app.route('/send_mail')
def send_mail():
    try:
        overall_file = 'overall_attendance.csv'
        student_file = 'studentdetails.csv'
        
        if not os.path.exists(overall_file):
            flash('Overall attendance data not found', 'error')
            return render_template('Email_send.html', students=[], errors=['Overall attendance data not found'])
        
        if not os.path.exists(student_file):
            flash('Student details not found', 'error')
            return render_template('Email_send.html', students=[], errors=['Student details not found'])

        # Read data
        attendance_df = pd.read_csv(overall_file)
        student_df = pd.read_csv(student_file)
        
        sent_students = []
        errors = []
        
        # Find students with attendance < 75%
        low_attendance = attendance_df[attendance_df['Percentage'] < 75]
        
        for _, attendance_row in low_attendance.iterrows():
            enrollment = str(attendance_row['Enrollment'])
            percentage = attendance_row['Percentage']
            
            # Find student details
            student_info = student_df[student_df['Moodle ID'].astype(str) == enrollment]
            
            if not student_info.empty:
                student_data = student_info.iloc[0]
                name = student_data['Name']
                email = student_data['Email ID']
                parent_email = student_data["Parent's Email"]
                
                # Send emails
                student_email_sent = send_email_to_student(email, name)
                parent_email_sent = send_email_to_parent(parent_email, name)
                
                if student_email_sent or parent_email_sent:
                    sent_students.append({
                        'name': name,
                        'enrollment': enrollment,
                        'percentage': percentage,
                        'student_email_sent': student_email_sent,
                        'parent_email_sent': parent_email_sent
                    })
                else:
                    errors.append(f"Failed to send emails for {name}")
        
        return render_template('Email_send.html', students=sent_students, errors=errors)
        
    except Exception as e:
        app.logger.error(f"Error in send_mail: {str(e)}")
        flash('Error processing email notifications', 'error')
        return render_template('Email_send.html', students=[], errors=[str(e)])

############################## ERROR HANDLERS ################################
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('StudentDetails', exist_ok=True)
    os.makedirs('Attendance_management/Attendance/spcc', exist_ok=True)
    os.makedirs('Attendance_management/Attendance/css', exist_ok=True)
    os.makedirs('Attendance_management/Attendance/ai', exist_ok=True)
    os.makedirs('Attendance_management/Attendance/mc', exist_ok=True)
    
    app.run(debug=True, host='127.0.0.1', port=5000)