# Clinical validation plan

Good results on a Kaggle leaderboard mean little on their own. Validation should
prove the system works **on the cameras, technicians and patients where it will
be deployed**. The plan has three stages, each with a clear go/no-go gate.

## Stage 1: Retrospective external validation (months 2–4)

- **Data:** at least 2 datasets the model never trained on. For example,
  Messidor-2 (with adjudicated grades), IDRiD (Indian patients), and about 2,000
  de-identified archived images from the partner hospital, taken on the target camera.
- **Reference standard:** ICDR grade from 2 independent retina specialists, with a
  third specialist adjudicating disagreements. Grade macular oedema separately
  (hard exudates within 1 disc diameter of the fovea).
- **Primary endpoint:** sensitivity and specificity for referable DR (moderate NPDR or
  worse, or macular oedema), with 95% CIs: `python -m drscreen.evaluate`.
- **Gate:** sensitivity ≥ 85% and specificity ≥ 80%, with the lower 95% CI bound
  of sensitivity ≥ 80%. This is broadly in line with the bar set by published
  pivotal trials of autonomous DR AI. Confirm the exact target with your clinical partners.
- **Subgroups:** camera model, age band, sex, image quality, dilated vs non-dilated
  pupils, and cataract. Report every subgroup, including weak ones.

## Stage 2: Prospective silent-mode pilot (months 5–8)

- **Setting:** 2–3 primary health centres (PHCs) or diabetes clinics. Trained
  technicians capture images, and the AI runs in the background. The AI result is
  **not** used for patient care.
- **Reference:** every patient also gets a specialist grading of the same images,
  plus a dilated exam by an ophthalmologist for a random 10% and for all
  discordant cases.
- **Sample size:** with referable DR prevalence around 10–15%, about 1,000 patients
  gives 100–150 referable cases. That estimates sensitivity of about 90% to within
  ±5–6% (95% CI). Recalculate with your local prevalence.
- **Also measure:** ungradable rate (target < 10%), time per patient, technician
  retake rate, and how often the system is down.
- **Approvals:** Institutional Ethics Committee approval, registration on CTRI
  (Clinical Trials Registry – India) before the first patient is enrolled, and
  written informed consent in the local language.

## Stage 3: Interventional pilot (months 9–14)

- **Setting:** AI-assisted referral pathway, with an ophthalmologist reviewing all
  AI-positive cases and a 10% random sample of AI-negative cases.
- **Outcomes:** referral completion rate, time from screening to treatment, and
  patients screened per technician-day. Track cost per patient screened and cost
  per referable case detected, because these numbers win government and CSR funding.

## Things that commonly invalidate DR AI studies

- The same patient's eyes ending up in both training and test sets. The code uses
  grouped splits by `patient_id`.
- Tuning the operating threshold on the test set. Tune it on validation and freeze it
  before running the test.
- Dropping ungradable images from the analysis. Report them, and count them as referrals.
- Spectrum bias, where test sets are enriched with severe disease. Report results at
  the real screening prevalence.
- Grading only for DR severity and ignoring macular oedema. Macular oedema is a
  leading cause of vision loss and must be part of "referable".

## Monitoring after launch

- Every week, review a random 2–5% of AI-negative cases alongside all AI-positive cases.
- Track the drift of the referable rate and ungradable rate per site and per camera.
- Lock model versions. Any retrained model goes through regulatory change control
  before it is used.
