# Project roadmap: from code to a screening programme

| Phase | Months | Key deliverables | Gate to next phase |
| --- | --- | --- | --- |
| 0. Partnerships | 0–2 | MoU with an eye hospital or medical college, a retina-specialist co-investigator, and a camera choice | Data access signed |
| 1. Model v1 | 1–4 | Trained on public data plus archived partner images; external validation report | Referable-DR sensitivity ≥ 85% and specificity ≥ 80% on external data |
| 2. Silent pilot | 5–8 | Ethics approval, CTRI registration, 2–3 PHC sites, about 1,000 patients | Sensitivity holds prospectively; ungradable rate < 10% |
| 3. Regulatory | 6–14 | QMS (ISO 13485 / IEC 62304 / ISO 14971), Class C SaMD dossier | CDSCO licence |
| 4. Interventional pilot | 9–14 | AI-assisted referral pathway, referral completion tracking, cost per case | Programme partner commits funding |
| 5. Scale | 15+ | State NCD programme, hospital chains, diabetology clinics, optical chains | Unit economics positive |

## Workstreams and owners

- **ML:** data pipeline, model, validation and monitoring (you, plus 1 ML engineer)
- **Clinical:** grading protocol, adjudication and pilot oversight (retina specialist)
- **Product:** technician app, offline mode, local languages and printing (1 full-stack engineer)
- **Regulatory and quality:** QMS, risk file and CDSCO dossier (consultant)
- **Field operations:** site onboarding, technician training and camera logistics (1 coordinator)

## Technical backlog (in priority order)

1. **Macular oedema head:** add an exudates-near-fovea output. Referable = DR ≥ moderate **or** macular oedema.
2. **Model ensemble and test-time augmentation:** typically adds 1–2 points of AUC.
3. **Camera-specific calibration of the quality gate,** using a few hundred images per device.
4. **Offline-first:** ONNX export and int8 quantisation so it runs on a ₹30k laptop or an Android tablet. Queue uploads for when there's no connectivity.
5. **Reviewer console:** an ophthalmologist confirms or overrides each result, and the overrides become new training labels.
6. **Monitoring dashboard:** referable rate, ungradable rate and drift per site.
7. **ABDM / HMIS integration** for government programmes.

## Funding routes to explore

- BIRAC BIG (Biotechnology Ignition Grant) for early proof of concept
- ICMR extramural grants and the Medical Device and Diagnostics Mission Secretariat
- State health missions (NCD programmes) for deployment
- Corporate CSR from pharma and diabetes-care companies
- Global health funds and foundations focused on preventable blindness

## Business model options

- **Per-screening fee** (for example ₹50–150 per patient) for private clinics and optical chains
- **Annual site licence** for hospitals and diabetology chains
- **Government tender** for state or district screening programmes
- **Bundling with camera makers** of low-cost fundus cameras
