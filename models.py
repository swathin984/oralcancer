import datetime as dt
from typing import Optional

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin


db = SQLAlchemy()


class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, index=True)  # 'admin' or 'doctor'
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    # Relationships
    patients = db.relationship("Patient", backref="doctor", lazy=True, cascade="all, delete-orphan")
    predictions = db.relationship("Prediction", backref="doctor", lazy=True, cascade="all, delete-orphan")

    def get_id(self) -> str:
        return str(self.id)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_doctor(self) -> bool:
        return self.role == "doctor"


class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    predictions = db.relationship("Prediction", backref="patient", lazy=True, cascade="all, delete-orphan", order_by="Prediction.created_at.desc()")

    def get_risk_level(self) -> str:
        """Calculate risk level based on prediction history"""
        if not self.predictions:
            return "unknown"
        
        recent_predictions = self.predictions[:3]  # Last 3 predictions
        malignant_count = sum(1 for p in recent_predictions if 'malignant' in p.predicted_class.lower())
        
        if malignant_count >= 2:
            return "high"
        elif malignant_count == 1:
            return "moderate"
        else:
            return "low"
    
    def get_risk_trend(self) -> str:
        """Analyze if risk is increasing, decreasing, or stable"""
        if len(self.predictions) < 2:
            return "insufficient_data"
        
        recent = self.predictions[:2]
        current_malignant = 'malignant' in recent[0].predicted_class.lower()
        previous_malignant = 'malignant' in recent[1].predicted_class.lower()
        
        if current_malignant and not previous_malignant:
            return "increasing"
        elif not current_malignant and previous_malignant:
            return "decreasing"
        else:
            return "stable"


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    original_image_path = db.Column(db.String(512), nullable=False)
    annotated_image_path = db.Column(db.String(512), nullable=False)
    predicted_class = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    notes = db.Column(db.Text, nullable=True)  # Doctor's notes about this prediction
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.2f}%"
    
    def get_risk_category(self) -> str:
        """Categorize the prediction risk"""
        if 'malignant' in self.predicted_class.lower():
            if self.confidence > 0.8:
                return "high_risk"
            else:
                return "moderate_risk"
        else:
            return "low_risk"
    
    def get_confidence_level(self) -> str:
        """Get confidence level description"""
        if self.confidence >= 0.9:
            return "very_high"
        elif self.confidence >= 0.7:
            return "high"
        elif self.confidence >= 0.5:
            return "moderate"
        else:
            return "low"