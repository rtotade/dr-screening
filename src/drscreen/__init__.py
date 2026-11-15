"""drscreen: AI-assisted diabetic retinopathy screening (research use only)."""

GRADES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
NUM_CLASSES = len(GRADES)
REFERABLE_MIN_GRADE = 2  # ICDR: moderate NPDR or worse is referable

__version__ = "0.1.0"
