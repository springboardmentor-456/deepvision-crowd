import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import io
import os
import time
import threading

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256
DOWNSAMPLE_FACTOR = 8
OUTPUT_SIZE = IMG_HEIGHT // DOWNSAMPLE_FACTOR

# SMTP Configuration Class
class SMTPConfig:
    def __init__(self):
        self.smtp_server = ""
        self.smtp_port = 587
        self.sender_email = ""
        self.sender_password = ""
        self.recipient_emails = []
        self.enabled = False

# Email Alert System
class EmailAlertSystem:
    def __init__(self, config: SMTPConfig):
        self.config = config
        self.last_alert_time = {}  # Track last alert time per location
        self.alert_cooldown = 300  # 5 minutes cooldown between alerts

    def send_alert_email(self, crowd_count, location, crowd_limit, density_map_image=None):
        """Send email alert for crowd limit breach"""
        if not self.config.enabled or not self.config.recipient_emails:
            return False, "SMTP not configured or no recipients"

        # Check cooldown
        current_time = time.time()
        last_alert = self.last_alert_time.get(location, 0)
        if current_time - last_alert < self.alert_cooldown:
            return False, f"Alert cooldown active for {location}"

        try:
            # Create message
            msg = MIMEMultipart('related')
            msg['From'] = self.config.sender_email
            msg['To'] = ', '.join(self.config.recipient_emails)
            msg['Subject'] = f"🚨 CROWD ALERT: {location} - {crowd_count} people detected"

            # Create HTML body
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; border: 2px solid #ff4444; border-radius: 10px; padding: 20px; background-color: #fff8f8;">
                        <h2 style="color: #ff4444; text-align: center; margin-bottom: 30px;">
                            🚨 CROWD LIMIT EXCEEDED
                        </h2>
                        
                        <div style="background-color: #fff; padding: 20px; border-radius: 5px; margin: 15px 0;">
                            <h3 style="color: #ff4444; margin-top: 0;">Alert Details:</h3>
                            <ul style="list-style-type: none; padding: 0;">
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Location:</strong> {location}
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Current Count:</strong> {crowd_count} people
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Crowd Limit:</strong> {crowd_limit} people
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Exceeded by:</strong> {crowd_count - crowd_limit} people ({((crowd_count/crowd_limit-1)*100):.1f}%)
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                </li>
                            </ul>
                        </div>
                        
                        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0;">
                            <h4 style="color: #856404; margin-top: 0;">⚠️ Recommended Actions:</h4>
                            <ul style="color: #856404;">
                                <li>Implement crowd control measures immediately</li>
                                <li>Consider temporarily restricting entry</li>
                                <li>Ensure adequate safety personnel are present</li>
                                <li>Monitor the situation continuously</li>
                            </ul>
                        </div>
                        
                        <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
                            <p style="color: #666; font-size: 12px;">
                                This is an automated alert from the Crowd Monitoring System<br>
                                Powered by CSRNet Deep Learning Model
                            </p>
                        </div>
                    </div>
                </body>
            </html>
            """

            # Attach HTML body
            msg.attach(MIMEText(html_body, 'html'))

            # Attach density map image if provided
            if density_map_image is not None:
                img_attachment = MIMEImage(density_map_image)
                img_attachment.add_header('Content-Disposition', 'attachment', filename='density_map.png')
                msg.attach(img_attachment)

            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.send_message(msg)

            # Update last alert time
            self.last_alert_time[location] = current_time
            return True, "Alert sent successfully"

        except Exception as e:
            return False, f"Failed to send alert: {str(e)}"

    def test_connection(self):
        """Test SMTP connection"""
        try:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                return True, "SMTP connection successful"
        except Exception as e:
            return False, f"SMTP connection failed: {str(e)}"

# CSRNet Model Definition
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33]) 
        
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# Load model
@st.cache_resource
def load_model():
    """Load the trained CSRNet model"""
    model = CSRNet().to(DEVICE)
    try:
        # Try to load pretrained weights
        #model.load_state_dict(torch.load("csrnet_partA_final.pth", map_location=DEVICE))
        model.load_state_dict(torch.load("csrnet_partA.pth", map_location=DEVICE))
        #model.load_state_dict(torch.load("crowd_counting_model_best.pth", map_location=DEVICE))
        st.success("✅ Loaded pretrained CSRNet model")
    except:
        st.warning("⚠️ No pretrained model found. Using randomly initialized weights.")
    return model

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Resize image
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
    
    return img_tensor, img_resized

# Crowd counting prediction
def predict_crowd_count(model, image):
    """Predict crowd count from image"""
    img_tensor, img_resized = preprocess_image(image)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        crowd_count = int(output.sum().item())
        density_map = output.squeeze().cpu().numpy()
    
    return crowd_count, density_map, img_resized

# Visualization functions
def create_density_visualization(original_img, density_map):
    """Create side-by-side visualization of original image and density map"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(original_img)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Density map
    im = ax2.imshow(density_map, cmap='jet', alpha=0.8)
    ax2.set_title("Predicted Density Map", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def create_crowd_gauge(count, limit):
    """Create a gauge chart for crowd count"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = count,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Crowd Count"},
        delta = {'reference': limit},
        gauge = {
            'axis': {'range': [None, limit * 2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, limit], 'color': "lightgray"},
                {'range': [limit, limit * 2], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': limit
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def save_density_map_image(fig):
    """Save density map visualization as image bytes"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

# Initialize session state
if 'crowd_history' not in st.session_state:
    st.session_state.crowd_history = []

if 'smtp_config' not in st.session_state:
    st.session_state.smtp_config = SMTPConfig()

if 'email_system' not in st.session_state:
    st.session_state.email_system = EmailAlertSystem(st.session_state.smtp_config)

# Streamlit App
def main():
    st.set_page_config(
        page_title="Crowd Monitoring Dashboard",
        page_icon="👥",
        layout="wide"
    )
    
    st.title("🚨 Crowd Monitoring Dashboard")
    st.markdown("Real-time crowd counting and alert system using CSRNet")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # SMTP Configuration Section
    with st.sidebar.expander("📧 Email Alert Settings", expanded=False):
        st.subheader("SMTP Configuration")
        
        # SMTP Settings
        smtp_server = st.text_input("SMTP Server", 
                                  value=st.session_state.smtp_config.smtp_server,
                                  placeholder="smtp.gmail.com")
        
        smtp_port = st.number_input("SMTP Port", 
                                  value=st.session_state.smtp_config.smtp_port,
                                  min_value=1, max_value=65535)
        
        sender_email = st.text_input("Sender Email", 
                                   value=st.session_state.smtp_config.sender_email,
                                   placeholder="your-email@gmail.com")
        
        sender_password = st.text_input("Sender Password", 
                                      type="password",
                                      placeholder="Enter app password")
        
        # Recipients
        st.subheader("Alert Recipients")
        recipient_input = st.text_area("Recipient Emails (one per line)",
                                     value='\n'.join(st.session_state.smtp_config.recipient_emails),
                                     placeholder="admin@company.com\nsecurity@company.com")
        
        # Enable/Disable alerts
        alerts_enabled = st.checkbox("Enable Email Alerts", 
                                   value=st.session_state.smtp_config.enabled)
        
        # Update configuration
        if st.button("💾 Save SMTP Settings"):
            st.session_state.smtp_config.smtp_server = smtp_server
            st.session_state.smtp_config.smtp_port = smtp_port
            st.session_state.smtp_config.sender_email = sender_email
            if sender_password:  # Only update if password is provided
                st.session_state.smtp_config.sender_password = sender_password
            st.session_state.smtp_config.recipient_emails = [email.strip() for email in recipient_input.split('\n') if email.strip()]
            st.session_state.smtp_config.enabled = alerts_enabled
            
            # Update email system
            st.session_state.email_system = EmailAlertSystem(st.session_state.smtp_config)
            st.success("SMTP settings saved!")
        
        # Test connection
        if st.button("🔧 Test SMTP Connection"):
            if st.session_state.smtp_config.enabled and st.session_state.smtp_config.smtp_server:
                with st.spinner("Testing SMTP connection..."):
                    success, message = st.session_state.email_system.test_connection()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("Please configure SMTP settings first")
        
        # Send test alert
        if st.button("📧 Send Test Alert"):
            if st.session_state.smtp_config.enabled and st.session_state.smtp_config.recipient_emails:
                with st.spinner("Sending test alert..."):
                    success, message = st.session_state.email_system.send_alert_email(
                        crowd_count=75,
                        location="Test Location",
                        crowd_limit=50
                    )
                if success:
                    st.success(f"✅ Test alert sent successfully")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("Please configure SMTP settings and recipients first")
    
    # Crowd limit setting
    crowd_limit = st.sidebar.slider("Crowd Limit", min_value=10, max_value=200, value=50, step=5)
    
    # Location setting
    location = st.sidebar.text_input("Location Name", value="Monitoring Area")
    
    # Alert settings
    with st.sidebar.expander("🚨 Alert Settings", expanded=False):
        auto_alerts = st.checkbox("Automatic Email Alerts", value=True)
        alert_threshold = st.slider("Alert Threshold (%)", min_value=0, max_value=50, value=0, step=5,
                                  help="Send alert when crowd exceeds limit by this percentage")
    
    # Load model
    model = load_model()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📸 Upload Image for Analysis")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Predict crowd count
            with st.spinner("Analyzing crowd density..."):
                crowd_count, density_map, processed_img = predict_crowd_count(model, image)
            
            # Check if alert should be triggered
            effective_limit = crowd_limit + (crowd_limit * alert_threshold / 100)
            should_alert = crowd_count > effective_limit
            
            # Store in history
            timestamp = datetime.now()
            st.session_state.crowd_history.append({
                'timestamp': timestamp,
                'count': crowd_count,
                'location': location,
                'is_alert': should_alert,
                'limit': crowd_limit,
                'threshold': effective_limit
            })
            
            # Send email alert if conditions are met
            if should_alert and auto_alerts and st.session_state.smtp_config.enabled:
                # Create density map image for attachment
                fig = create_density_visualization(processed_img, density_map)
                density_image = save_density_map_image(fig)
                
                # Send alert in background thread to avoid blocking UI
                def send_alert():
                    success, message = st.session_state.email_system.send_alert_email(
                        crowd_count=crowd_count,
                        location=location,
                        crowd_limit=crowd_limit,
                        density_map_image=density_image
                    )
                    return success, message
                
                # Execute alert
                with st.spinner("Sending alert email..."):
                    alert_success, alert_message = send_alert()
                
                if alert_success:
                    st.success("📧 Alert email sent successfully!")
                else:
                    st.warning(f"⚠️ Alert email failed: {alert_message}")
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            # Status indicator
            if should_alert:
                st.error(f"🚨 ALERT: Overcrowded! Count: {crowd_count} (Effective Limit: {int(effective_limit)})")
                alert_status = True
            else:
                st.success(f"✅ Normal crowd level. Count: {crowd_count} (Limit: {crowd_limit})")
                alert_status = False
            
            # Create visualization
            fig = create_density_visualization(processed_img, density_map)
            st.pyplot(fig)
            
            # Show alert notification and recommendations
            if alert_status:
                st.warning("⚠️ Consider implementing crowd control measures!")
                
                with st.expander("📋 Recommended Actions", expanded=True):
                    st.markdown("""
                    ### Immediate Actions Required:
                    - 🚫 **Temporarily restrict entry** to the area
                    - 👮 **Deploy additional security personnel** 
                    - 📢 **Activate crowd control measures**
                    - 📊 **Continuous monitoring** of the situation
                    - 🚨 **Prepare emergency response** if needed
                    
                    ### Safety Considerations:
                    - Ensure emergency exits are clear and accessible
                    - Monitor crowd movement patterns
                    - Have medical personnel on standby
                    - Coordinate with local authorities if necessary
                    """)
    
    with col2:
        st.header("📈 Live Dashboard")
        
        # Email alert status
        if st.session_state.smtp_config.enabled:
            st.success("📧 Email alerts: ENABLED")
            if st.session_state.smtp_config.recipient_emails:
                st.info(f"📬 Recipients: {len(st.session_state.smtp_config.recipient_emails)}")
        else:
            st.warning("📧 Email alerts: DISABLED")
        
        # Crowd gauge
        if uploaded_file is not None:
            gauge_fig = create_crowd_gauge(crowd_count, crowd_limit)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Statistics")
        if st.session_state.crowd_history:
            df = pd.DataFrame(st.session_state.crowd_history)
            
            # Current stats
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.metric("Current Count", crowd_count if uploaded_file else 0)
                st.metric("Total Alerts", len(df[df['is_alert']]))
            
            with col_stats2:
                st.metric("Crowd Limit", crowd_limit)
                st.metric("Total Scans", len(df))
            
            # History chart
            if len(df) > 1:
                st.subheader("📈 Crowd History")
                
                # Create time series chart
                fig_history = px.line(df, x='timestamp', y='count', 
                                    title='Crowd Count Over Time',
                                    color_discrete_sequence=['blue'])
                fig_history.add_hline(y=crowd_limit, line_dash="dash", 
                                    line_color="red", annotation_text="Crowd Limit")
                fig_history.update_layout(height=300)
                st.plotly_chart(fig_history, use_container_width=True)
        
        # Recent alerts
        st.subheader("🚨 Recent Alerts")
        if st.session_state.crowd_history:
            recent_alerts = [h for h in st.session_state.crowd_history[-10:] if h['is_alert']]
            if recent_alerts:
                for alert in reversed(recent_alerts[-5:]):  # Show last 5 alerts
                    st.error(f"⚠️ {alert['timestamp'].strftime('%H:%M:%S')}: {alert['count']} people at {alert['location']}")
            else:
                st.info("No recent alerts")
        
        # Alert summary
        if st.session_state.crowd_history:
            df = pd.DataFrame(st.session_state.crowd_history)
            if len(df[df['is_alert']]) > 0:
                st.subheader("📊 Alert Summary")
                alert_df = df[df['is_alert']].copy()
                avg_count = alert_df['count'].mean()
                max_count = alert_df['count'].max()
                total_alerts = len(alert_df)
                
                col_alert1, col_alert2 = st.columns(2)
                with col_alert1:
                    st.metric("Avg Alert Count", f"{avg_count:.0f}")
                with col_alert2:
                    st.metric("Max Alert Count", f"{max_count}")
        
        # System status
        st.subheader("🖥️ System Status")
        gpu_status = "🟢 GPU Available" if torch.cuda.is_available() else "🟡 CPU Only"
        st.info(gpu_status)
        
        device_info = f"Device: {DEVICE}"
        st.info(device_info)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.crowd_history = []
            st.success("History cleared!")
            st.rerun()
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2 = st.columns(2)
    with col_footer1:
        st.markdown("*Powered by CSRNet deep learning model for accurate crowd counting*")
    with col_footer2:
        if st.session_state.smtp_config.enabled:
            st.markdown("*📧 Email alerts active*")
        else:
            st.markdown("*📧 Email alerts inactive*")

if __name__ == "__main__":
    main()