import os
import uuid
import cv2
from datetime import timedelta, datetime
import pytz
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, session, abort
from werkzeug.utils import secure_filename
from inference import get_model
import supervision as sv
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt

from models import db, User, Patient, Prediction


app = Flask(__name__)

# Basic Configuration
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///oral_cancer_classifier.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=7)

# File Uploads
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Init extensions
bcrypt = Bcrypt(app)
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Template filter to convert UTC to local time
@app.template_filter('localtime')
def localtime(utc_dt):
    """Convert UTC datetime to local timezone for display"""
    if utc_dt is None:
        return None
    
    # If datetime is naive, assume it's UTC
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    
    # Get system local timezone
    try:
        import time
        # Get the current UTC offset in seconds
        utc_offset = -time.timezone if (time.daylight == 0) else -time.altzone
        offset_hours = utc_offset // 3600
        offset_minutes = (utc_offset % 3600) // 60
        
        # Convert UTC to local time
        local_dt = utc_dt + timedelta(hours=offset_hours, minutes=offset_minutes)
        return local_dt.replace(tzinfo=None)
    except:
        # Fallback: just return the UTC time
        return utc_dt.replace(tzinfo=None)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('doctor_dashboard'))
    return render_template('landing.html')


# Authentication
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not name or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('signup.html')
        pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(name=name, email=email, password_hash=pw_hash, role='doctor')
        db.session.add(user)
        db.session.commit()
        flash('Account created. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, password) and user.is_active:
            login_user(user, remember=True)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('admin_dashboard' if user.is_admin else 'doctor_dashboard'))
        flash('Invalid credentials.', 'error')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'success')
    return redirect(url_for('home'))


# Doctor routes
@app.route('/doctor')
@login_required
def doctor_dashboard():
    if not current_user.is_doctor:
        return abort(403)
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.created_at.desc()).all()
    recents = Prediction.query.filter_by(doctor_id=current_user.id).order_by(Prediction.created_at.desc()).limit(10).all()
    return render_template('doctor/dashboard.html', patients=patients, predictions=recents)


@app.route('/doctor/patient/new', methods=['GET', 'POST'])
@login_required
def new_patient():
    if not current_user.is_doctor:
        return abort(403)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        history = request.form.get('medical_history', '').strip()
        if not name:
            flash('Patient name is required.', 'error')
            return render_template('doctor/new_patient.html')
        try:
            age_val = int(age) if age else None
        except ValueError:
            flash('Age must be a number.', 'error')
            return render_template('doctor/new_patient.html')
        patient = Patient(doctor_id=current_user.id, name=name, age=age_val, gender=gender, medical_history=history)
        db.session.add(patient)
        db.session.commit()
        flash('Patient created successfully! Now upload an image for analysis.', 'success')
        # Redirect directly to predict for this new patient
        return redirect(url_for('doctor_predict', patient_id=patient.id))
    return render_template('doctor/new_patient.html')


@app.route('/doctor/patient/<int:patient_id>/timeline')
@login_required
def patient_timeline(patient_id):
    if not current_user.is_doctor:
        return abort(403)
    patient = Patient.query.filter_by(id=patient_id, doctor_id=current_user.id).first_or_404()
    predictions = patient.predictions  # Already ordered by created_at desc due to model relationship
    return render_template('doctor/patient_timeline.html', patient=patient, predictions=predictions)


@app.route('/doctor/prediction/<int:prediction_id>/notes', methods=['POST'])
@login_required
def update_prediction_notes(prediction_id):
    if not current_user.is_doctor:
        return abort(403)
    prediction = Prediction.query.filter_by(id=prediction_id, doctor_id=current_user.id).first_or_404()
    notes = request.form.get('notes', '').strip()
    prediction.notes = notes
    db.session.commit()
    flash('Notes updated successfully.', 'success')
    return redirect(url_for('prediction_result', prediction_id=prediction.id))


@app.route('/doctor/predict/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def doctor_predict(patient_id):
    if not current_user.is_doctor:
        return abort(403)
    patient = Patient.query.filter_by(id=patient_id, doctor_id=current_user.id).first_or_404()
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'error')
            return render_template('doctor/predict.html', patient=patient)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return render_template('doctor/predict.html', patient=patient)
        if file and allowed_file(file.filename):
            # secure and unique filename
            ext = file.filename.rsplit('.', 1)[1].lower()
            base = secure_filename(file.filename.rsplit('.', 1)[0])
            filename = f"{base}_{uuid.uuid4().hex[:8]}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            api_key = os.getenv('ROBOFLOW_API_KEY')
            if not api_key:
                flash('ROBOFLOW_API_KEY not set.', 'error')
                return render_template('doctor/predict.html', patient=patient)
            try:
                image = cv2.imread(filepath)
                if image is None:
                    flash('Could not load image.', 'error')
                    return render_template('doctor/predict.html', patient=patient)

                model = get_model(model_id="oral-cancer-tsvhr/1", api_key=api_key)

                # The inference SDK returns an object or dict depending on version; normalize
                results = model.infer(image)[0]
                predicted_class = None
                confidence = 0.0
                # Try attribute-style first (older SDK wrappers)
                if hasattr(results, 'predicted_classes') and hasattr(results, 'predictions'):
                    predicted_class = results.predicted_classes[0] if results.predicted_classes else 'Unknown'
                    preds = results.predictions
                    if predicted_class in preds:
                        confidence = getattr(preds[predicted_class], 'confidence', 0.0) or 0.0
                else:
                    # dict-like path (modern HTTP client usually returns dict)
                    # Expected format: { "top": [ {"class": "...", "confidence": 0.97}, ... ] }
                    if isinstance(results, dict):
                        top = results.get('top') or results.get('predictions') or []
                        if isinstance(top, list) and top:
                            first = top[0]
                            predicted_class = first.get('class', 'Unknown')
                            confidence = float(first.get('confidence', 0.0))
                if not predicted_class:
                    predicted_class = 'Unknown'

                # Annotate the image
                display = image.copy()
                text = f"{predicted_class}: {confidence:.2%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(display, (5, 5), (int(5 + 10 + 10 + 10 + max(200, len(text) * 15)), 50), (255, 255, 255), -1)
                cv2.putText(display, text, (10, 40), font, 1, (0, 128, 0), 2, cv2.LINE_AA)

                output_filename = f"annotated_{filename}"
                output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                cv2.imwrite(output_filepath, display)

                # Save prediction record
                pred = Prediction(
                    doctor_id=current_user.id,
                    patient_id=patient.id,
                    original_image_path=filename,
                    annotated_image_path=output_filename,
                    predicted_class=predicted_class,
                    confidence=float(confidence),
                )
                db.session.add(pred)
                db.session.commit()

                # Redirect to dedicated result page
                flash('Analysis completed successfully!', 'success')
                return redirect(url_for('prediction_result', prediction_id=pred.id))

            except Exception as e:
                flash(f"Error processing image: {str(e)}", 'error')
        else:
            flash('Please upload a valid image file (PNG, JPG, JPEG).', 'error')
    
    return render_template('doctor/predict.html', patient=patient)


@app.route('/doctor/prediction/<int:prediction_id>/result')
@login_required
def prediction_result(prediction_id):
    if not current_user.is_doctor:
        return abort(403)
    prediction = Prediction.query.filter_by(id=prediction_id, doctor_id=current_user.id).first_or_404()
    patient = prediction.patient
    # Get other predictions for comparison
    other_predictions = Patient.query.get(patient.id).predictions
    return render_template('doctor/prediction_result.html', 
                         prediction=prediction, 
                         patient=patient, 
                         other_predictions=other_predictions)


@app.route('/doctor/history')
@login_required
def doctor_history():
    if not current_user.is_doctor:
        return abort(403)
    preds = Prediction.query.filter_by(doctor_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('doctor/history.html', predictions=preds)


# PDF report generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def generate_pdf_report(pred: Prediction, patient: Patient, output_path: str):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 50

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, "Oral Cancer Classification Report")
    c.setFont("Helvetica", 10)
    c.drawString(margin, height - margin - 15, f"Date: {pred.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    # Patient details
    y = height - margin - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Patient Details")
    c.setFont("Helvetica", 11)
    y -= 18
    c.drawString(margin, y, f"Name: {patient.name}")
    y -= 16
    if patient.age is not None:
        c.drawString(margin, y, f"Age: {patient.age}")
        y -= 16
    if patient.gender:
        c.drawString(margin, y, f"Gender: {patient.gender}")
        y -= 16
    if patient.medical_history:
        c.drawString(margin, y, "Medical History:")
        y -= 14
        # Basic wrap
        text_obj = c.beginText(margin, y)
        text_obj.setFont("Helvetica", 10)
        for line in str(patient.medical_history).split('\n'):
            text_obj.textLine(line[:120])
        c.drawText(text_obj)
        y = text_obj.getY() - 10

    # Prediction details
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Prediction")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Class: {pred.predicted_class}")
    y -= 16
    c.drawString(margin, y, f"Confidence: {pred.confidence_pct()}")

    # Annotated image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], pred.annotated_image_path)
    try:
        img = ImageReader(img_path)
        # Fit image within page area
        max_w = width - 2 * margin
        max_h = height / 2
        iw, ih = img.getSize()
        scale = min(max_w / iw, max_h / ih)
        draw_w = iw * scale
        draw_h = ih * scale
        c.drawImage(img, margin, margin, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
    except Exception:
        c.setFont("Helvetica", 10)
        c.drawString(margin, margin + 10, "[Annotated image unavailable]")

    c.showPage()
    c.save()


@app.route('/doctor/report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    if not current_user.is_doctor:
        return abort(403)
    pred = Prediction.query.filter_by(id=prediction_id, doctor_id=current_user.id).first_or_404()
    patient = pred.patient
    reports_dir = os.path.join('static', 'uploads')
    os.makedirs(reports_dir, exist_ok=True)
    pdf_name = f"report_{pred.id}.pdf"
    pdf_path = os.path.join(reports_dir, pdf_name)
    generate_pdf_report(pred, patient, pdf_path)
    return send_from_directory(reports_dir, pdf_name, as_attachment=True)


# Admin routes
def require_admin():
    if not current_user.is_authenticated or not current_user.is_admin:
        abort(403)


@app.route('/admin')
@login_required
def admin_dashboard():
    require_admin()
    doctors = User.query.filter_by(role='doctor').order_by(User.created_at.desc()).all()
    recent_preds = Prediction.query.order_by(Prediction.created_at.desc()).limit(20).all()
    return render_template('admin/dashboard.html', doctors=doctors, predictions=recent_preds)


@app.route('/admin/doctor/new', methods=['GET', 'POST'])
@login_required
def admin_new_doctor():
    require_admin()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not all([name, email, password]):
            flash('All fields are required.', 'error')
            return render_template('admin/new_doctor.html')
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'error')
            return render_template('admin/new_doctor.html')
        pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(name=name, email=email, password_hash=pw_hash, role='doctor')
        db.session.add(user)
        db.session.commit()
        flash('Doctor created.', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/new_doctor.html')


@app.route('/admin/doctor/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
def admin_edit_doctor(user_id):
    require_admin()
    user = User.query.filter_by(id=user_id, role='doctor').first_or_404()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        is_active = request.form.get('is_active') == 'on'
        if not name or not email:
            flash('Name and email are required.', 'error')
            return render_template('admin/edit_doctor.html', user=user)
        # Check unique email
        existing = User.query.filter(User.email == email, User.id != user.id).first()
        if existing:
            flash('Email already in use.', 'error')
            return render_template('admin/edit_doctor.html', user=user)
        user.name = name
        user.email = email
        user.is_active = is_active
        db.session.commit()
        flash('Doctor updated.', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/edit_doctor.html', user=user)


@app.route('/admin/doctor/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_doctor(user_id):
    require_admin()
    user = User.query.filter_by(id=user_id, role='doctor').first_or_404()
    db.session.delete(user)
    db.session.commit()
    flash('Doctor deleted.', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/predictions')
@login_required
def admin_predictions():
    require_admin()
    preds = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template('admin/predictions.html', predictions=preds)


@app.route('/admin/prediction/<int:prediction_id>/delete', methods=['POST'])
@login_required
def admin_delete_prediction(prediction_id):
    require_admin()
    pred = Prediction.query.get_or_404(prediction_id)
    db.session.delete(pred)
    db.session.commit()
    flash('Prediction deleted.', 'success')
    return redirect(url_for('admin_predictions'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(403)
def forbidden(e):
    return render_template('errors/403.html'), 403


@app.errorhandler(404)
def not_found(e):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('errors/500.html'), 500


def ensure_admin_seed():
    # Seed a default admin if none exists and ADMIN_EMAIL/PASSWORD present
    admin_email = os.getenv('ADMIN_EMAIL')
    admin_password = os.getenv('ADMIN_PASSWORD')
    if admin_email and admin_password and not User.query.filter_by(email=admin_email).first():
        pw_hash = bcrypt.generate_password_hash(admin_password).decode('utf-8')
        admin = User(name='Admin', email=admin_email, password_hash=pw_hash, role='admin')
        db.session.add(admin)
        db.session.commit()


with app.app_context():
    db.create_all()
    ensure_admin_seed()


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
