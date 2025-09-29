import os
import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from inference import load_csrnet_model, get_count_and_heatmap
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

def send_alert_email(subject, to_email, overlay_img, plot_img, crowd_count, threshold, exceed_by, uploaded_filename):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["To"] = to_email
    msg["From"] = os.environ.get("SMTP_USER")
    msg.set_content("This is an HTML email. Please view in HTML capable client.")

    overlay_cid = make_msgid(domain="xyz.com")
    plot_cid = make_msgid(domain="xyz.com")

    buf_overlay = io.BytesIO()
    overlay_img.save(buf_overlay, format="PNG")
    buf_overlay.seek(0)

    buf_plot = io.BytesIO()
    plot_img.save(buf_plot, format="PNG")
    buf_plot.seek(0)

    html_content = f"""
    <html>
    <body>
        <h2 style="color:red;">🚨 Crowd Alert Notification</h2>
        <p><b>Uploaded Image:</b> {uploaded_filename}</p>
        <p><b>Estimated Crowd Count:</b> {crowd_count}</p>
        <p><b>Crowd exceeds the threshold of {threshold} by {exceed_by} people.</b></p>
        <p><b>Threshold:</b> {threshold}</p>
        <p><b>Status:</b> <span style='color:red;'>Crowd exceeds threshold!</span></p>
        <h3>Heatmap Overlay:</h3>
        <img src="cid:{overlay_cid[1:-1]}" width="600"><br><br>
        <h3>Crowd Comparison Plot:</h3>
        <img src="cid:{plot_cid[1:-1]}" width="600">
        <p>Please take necessary action.</p>
    </body>
    </html>
    """

    msg.add_alternative(html_content, subtype="html")
    msg.get_payload()[0].add_related(
        buf_overlay.getvalue(), maintype="image", subtype="png", cid=overlay_cid
    )
    msg.get_payload()[0].add_related(
        buf_plot.getvalue(), maintype="image", subtype="png", cid=plot_cid
    )

    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        print(f"[INFO] Alert sent to {to_email}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False

try:
    get_cache = st.cache_resource
except AttributeError:
    get_cache = lambda func: st.cache(allow_output_mutation=True)


@get_cache
def get_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "csrnet_model", "csrnet_train.pth")
    model = load_csrnet_model(MODEL_PATH)
    return model


model = get_model()

st.title("👥 Crowd Monitoring System")
st.write("Upload an image to estimate crowd count and visualize heatmap.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
CROWD_THRESHOLD = 100

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    st.image(img_pil, caption="Uploaded Image", use_column_width=True)

    overlay, count = get_count_and_heatmap(model, img_pil)

    if isinstance(overlay, np.ndarray):
        overlay = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    st.image(overlay, caption=f"Estimated Count: {count}", use_column_width=True)
    st.success(f"Estimated Crowd Count: {count}")

    if count > CROWD_THRESHOLD:
        exceed_by = count - CROWD_THRESHOLD
        st.warning(f"Crowd exceeds threshold ({CROWD_THRESHOLD})!")
        st.info(f"Crowd exceeds the threshold of {CROWD_THRESHOLD} by {exceed_by} people.")

        fig, ax = plt.subplots()
        ax.bar(["Threshold", "Estimated"], [CROWD_THRESHOLD, count], color=["red", "blue"])
        ax.set_ylabel("Crowd Count")
        ax.set_title("Crowd Alert Summary")
        buf_plot = io.BytesIO()
        fig.savefig(buf_plot, format="PNG")
        buf_plot.seek(0)
        plot_img = Image.open(buf_plot)

        if st.button("Send Alert Email"):
            success = send_alert_email(
                subject="🚨 Crowd Alert Notification",
                to_email="receiver@example.com",  # Change this to recipient email
                overlay_img=overlay,
                plot_img=plot_img,
                crowd_count=count,
                threshold=CROWD_THRESHOLD,
                exceed_by=exceed_by,  
                uploaded_filename=uploaded_file.name,
            )
            if success:
                st.success("✅ Alert email sent with heatmap and plot!")
            else:
                st.error("❌ Failed to send alert email. Check logs.")
