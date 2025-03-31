# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted or
declined, etc.

Data Protection and GDPR Compliance

The algorithm processes webcam video data, which may include personal and biometric information (e.g., facial landmarks, expressions). 
To comply with GDPR ethical guidelines, the system adheres to the following security measures:

Data Minimization: Only the necessary data (e.g., eye state, posture, expression) is processed to estimate mood and behavior, and raw video is not stored.
User Consent: Users must provide explicit consent before the system accesses the webcam, and they can stop the process at any time by pressing q.
Secure Processing: All data processing occurs locally on the user’s device, and no data is transmitted to external servers, reducing the risk of unauthorized access.
Output Security: Generated outputs (narrative.docx, mood_calculation.docx, visualizations.png, webcam_analysis.log) are stored locally and do not contain raw video frames, ensuring that sensitive biometric data is not exposed.
We are committed to regularly reviewing and updating our security practices to protect user data and maintain compliance with GDPR and other relevant regulations. 
If you have concerns about data privacy or security, please contact us at the email provided above.
