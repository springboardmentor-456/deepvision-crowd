DeepVision Crowd Monitoring System

DeepVision is an advanced crowd monitoring application designed to provide real-time density estimation, visual analytics, and smart alerts for large-scale gatherings. The system leverages deep learning to analyze video streams and provide actionable insights for safety, management, and research.

Key Features

- Real-time Crowd Estimation: Uses deep learning to generate density maps and count people in live video feeds or uploaded videos.
- Smart Alerts: Configurable notifications via email (SMTP/Gmail) and SMS (Twilio) when crowd thresholds are exceeded.
- Visual Analytics: Generates intuitive heatmaps and graphs to track crowd dynamics over time.
- Flexible Input: Supports MP4 video uploads and live camera streams.
- Customizable Processing: Frame interval adjustments for speed vs. accuracy trade-offs.

Core Components

File/Folder                         Description
Deepvision.py.py                  Main Python script containing the full processing logic, model inference, and alerting functions.
output_demo.pdf                   PDF documentation showing sample outputs, screenshots, and workflow.
requirements.txt                  List of Python dependencies required to run the project.
packages.txt                      Optional Linux package installation instructions.
README.md                         Project description, setup instructions, and usage guide.
LICENSE                           Licensing information for the project.

Installation and Setup

1. Clone the Repository (if using local machine):
   git clone <repository-url>
   cd deepvision-crowd

2. Install Python Dependencies:
   pip install -r requirements.txt

3. Set Environment Variables for Alerts (Optional):
   - Email Alerts:
     export SENDER_EMAIL="your_email@gmail.com"
     export SENDER_PASSWORD="app_password"
   - Twilio SMS Alerts:
     export TWILIO_ACCOUNT_SID="your_sid"
     export TWILIO_AUTH_TOKEN="your_token"
     export TWILIO_PHONE_NUMBER="+1234567890"

Usage

1. Run the main script:
   python Deepvision.py.py
2. Upload your video or connect a live stream.
3. Set crowd threshold and frame interval.
4. Enable alerts and configure recipient contacts.
5. Monitor real-time density map, crowd count, and alert status.

Output

- Real-time density heatmaps overlayed on video frames.
- Peak and average crowd count metrics.
- Email/SMS alerts triggered on threshold violations.

Project Workflow

1. Load and preprocess video input
2. Run deep learning model for crowd density estimation
3. Generate heatmaps and count metrics
4. Send alerts if thresholds are exceeded
5. Display results in real-time dashboard

License

This project is licensed under the MIT License – see the LICENSE file for details.

Contributing

Contributions, suggestions, and improvements are welcome. Please open an issue or submit a pull request.

Contact

For questions or support, reach out to the DeepVision Team at:
maanisha504@gmail.com
