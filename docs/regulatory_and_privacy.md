# Regulatory and privacy notes (India)

> These notes are for planning only and are not legal advice. Engage a CDSCO
> regulatory consultant and a data-protection lawyer before any clinical use.

## Medical device regulation (CDSCO)

- Standalone software with a medical purpose is regulated as **Software as a
  Medical Device (SaMD)** under the Medical Devices Rules, 2017.
- CDSCO's SaMD classification list puts a **"Diabetic Retinopathy Detection
  Device"** (adaptive-algorithm software that evaluates ophthalmic images for
  diagnostic screening) in **Class C (moderate-high risk)**. Plan for Class C from
  the start.
- CDSCO published final guidance on Medical Device Software, including expectations
  for AI/ML, in July 2026. For AI software it expects post-market monitoring for
  performance drift, bias, false outputs and safety signals, plus change control
  for model updates.
- **Precedent:** Forus Health received CDSCO approval for its AI DR screening tool
  as Class C SaMD in February 2026, so the pathway is proven and there are
  competitors already approved.

### What a Class C submission typically needs (to plan for, not an exhaustive list)
- A Quality Management System aligned to ISO 13485, with **IEC 62304** (software
  life cycle) and **ISO 14971** (risk management) processes
- Usability engineering (IEC 62366) covering the technician workflow
- Clinical performance evidence (see `clinical_validation_plan.md`)
- Cybersecurity documentation and a software bill of materials
- An intended-use statement, labelling, and an algorithm change protocol

### Fastest legal path while building evidence
Run the software as a **research tool** inside an ethics-approved study, where AI
outputs do not drive care in Stage 2. The device licence is needed before marketed
clinical use.

## Data protection (DPDP Act 2023 and DPDP Rules 2025)

- The DPDP Rules were notified in November 2025 and are being phased in, with full
  compliance due around May 2027. Design for full compliance now.
- **Notice and consent:** provide an itemised notice in the patient's language and
  get consent for screening, plus *separate* consent for using images in research
  or model training.
- **Security safeguards:** encryption at rest and in transit, access control, and
  logging.
- **Breaches:** notify the Data Protection Board within 72 hours, and inform
  affected people without delay.
- **Retention:** delete data when its purpose is complete. Document a retention
  schedule.

### What this codebase already does
- It stores only a pseudonymous `patient_ref`, with no name, phone number or Aadhaar.
- Raw images are not stored by default (`DR_STORE_IMAGES=0`).
- The model runs locally on a clinic PC, so images don't need to leave the site.

### Still to add before a pilot
- HTTPS, user login with roles (technician, reviewer, admin), and an audit log
- A consent-capture step in the UI and a retention/deletion job
- Image de-identification that strips EXIF/DICOM metadata at capture time
- ABDM integration (ABHA ID, health records) if you're joining public programmes

## Dataset licences
APTOS, EyePACS (Kaggle), IDRiD and Messidor-2 each have their own terms. Some
restrict commercial use. Check each licence, and for a commercial product rely
mainly on data from partner hospitals collected with consent.
