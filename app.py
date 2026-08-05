
import os
import smtplib
import uuid
from email.message import EmailMessage

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Travel Agent",
    page_icon="✈️",
    layout="centered",
)


# ============================================================
# CUSTOM CSS
# ============================================================

def render_custom_css():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.5rem;
            text-align: center;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .sub-title {
            font-size: 1.1rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .footer {
            text-align: center;
            margin-top: 2rem;
            font-size: 0.85rem;
            opacity: 0.7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GET SECRET
# ============================================================

def get_secret(name):
    """
    Get a secret from Streamlit Secrets first,
    then from environment variables.
    """

    try:
        value = st.secrets.get(name)

        if value:
            return value

    except Exception:
        pass

    return os.getenv(name)


# ============================================================
# GOOGLE API KEY
# ============================================================

def get_google_api_key():
    return get_secret("GOOGLE_API_KEY")


# ============================================================
# INITIALIZE GEMINI
# ============================================================

@st.cache_resource
def initialize_model(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )


# ============================================================
# GENERATE TRAVEL INFORMATION
# ============================================================

def generate_travel_information(user_query):

    api_key = get_google_api_key()

    if not api_key:
        raise ValueError(
            "Google API key not found. "
            "Please add GOOGLE_API_KEY to Streamlit Secrets."
        )

    llm = initialize_model(api_key)

    system_prompt = """
You are an AI Travel Agent.

Your job is to create useful, realistic and well-structured
travel information based on the user's request.

IMPORTANT RULES:

- Do not invent live flight prices.
- Do not invent live hotel availability.
- Do not claim that a flight or hotel is currently available
  unless the user provides that information.
- Clearly distinguish estimated information from confirmed
  information.
- If the user asks for flights or hotels, provide useful
  recommendations, areas, planning guidance and approximate
  budget information.
- Never fabricate booking confirmation numbers.
- Never fabricate real-time prices or schedules.

Structure the answer when appropriate using:

1. Trip Overview
2. Recommended Itinerary
3. Flights
4. Hotels
5. Transportation
6. Food Recommendations
7. Activities and Attractions
8. Estimated Budget
9. Travel Tips
10. Important Things to Check

Make the response practical, detailed and easy to read.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    response = llm.invoke(messages)

    return response.content


# ============================================================
# EMAIL SETTINGS
# ============================================================

def get_email_settings():

    return {
        "host": get_secret("SMTP_HOST"),
        "port": get_secret("SMTP_PORT"),
        "username": get_secret("SMTP_USERNAME"),
        "password": get_secret("SMTP_PASSWORD"),
        "use_tls": get_secret("SMTP_USE_TLS"),
    }


# ============================================================
# SEND EMAIL
# ============================================================

def send_email(
    sender_email,
    receiver_email,
    subject,
    travel_information,
):

    settings = get_email_settings()

    if not settings["host"]:
        raise ValueError("SMTP_HOST is not configured.")

    if not settings["port"]:
        raise ValueError("SMTP_PORT is not configured.")

    if not settings["username"]:
        raise ValueError("SMTP_USERNAME is not configured.")

    if not settings["password"]:
        raise ValueError("SMTP_PASSWORD is not configured.")

    try:
        port = int(settings["port"])

    except ValueError as exc:
        raise ValueError(
            "SMTP_PORT must be a number."
        ) from exc

    message = EmailMessage()

    message["Subject"] = subject
    message["From"] = settings["username"]
    message["To"] = receiver_email

    if sender_email:
        message["Reply-To"] = sender_email

    message.set_content(
        f"""
AI Travel Agent
===============

Travel Information

{travel_information}

-----------------------------------
Generated by AI Travel Agent
"""
    )

    use_tls = str(
        settings["use_tls"]
    ).lower() in {
        "true",
        "1",
        "yes",
    }

    with smtplib.SMTP(
        settings["host"],
        port,
        timeout=30,
    ) as server:

        if use_tls:
            server.starttls()

        server.login(
            settings["username"],
            settings["password"],
        )

        server.send_message(message)


# ============================================================
# HEADER
# ============================================================

def render_header():

    st.markdown(
        '<div class="main-title">'
        '✈️🌍 AI Travel Agent 🏨🗺️'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Enter your travel requirements and get an
        AI-generated travel plan with itinerary,
        flight guidance, hotel suggestions,
        activities and budget information.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.header("✈️ Travel Agent")

        st.write(
            """
            This AI assistant can help you plan:

            • Destinations
            • Itineraries
            • Flights
            • Hotels
            • Activities
            • Transportation
            • Food
            • Budget
            • Travel tips
            """
        )

        st.divider()

        st.info(
            """
            Flight and hotel prices/availability are not
            live unless a real-time travel API is connected.
            """
        )

        st.divider()

        st.caption(
            "AI Travel Agent • "
            "Streamlit + LangChain + Google Gemini"
        )


# ============================================================
# TRAVEL INPUT
# ============================================================

def render_travel_input():

    st.subheader("🌍 Plan Your Trip")

    user_input = st.text_area(
        "Travel Query",
        height=180,
        placeholder=(
            "Example:\n"
            "Plan a 7-day trip from Delhi to Dubai for "
            "2 people with a budget of ₹1,50,000. "
            "Include hotels, activities, transportation "
            "and a daily itinerary."
        ),
        key="travel_query",
    )

    return user_input


# ============================================================
# PROCESS QUERY
# ============================================================

def process_query(user_input):

    if not user_input or not user_input.strip():

        st.warning("Please enter a travel query.")

        return

    thread_id = str(uuid.uuid4())

    st.session_state.thread_id = thread_id

    with st.spinner("✈️ Planning your trip..."):

        try:

            travel_information = (
                generate_travel_information(
                    user_input.strip()
                )
            )

            st.session_state.travel_info = (
                travel_information
            )

        except Exception as exc:

            st.error(
                f"Unable to generate travel information: {exc}"
            )


# ============================================================
# DISPLAY TRAVEL INFORMATION
# ============================================================

def render_travel_information():

    if "travel_info" not in st.session_state:
        return

    st.divider()

    st.subheader("🗺️ Travel Information")

    st.markdown(
        st.session_state.travel_info
    )


# ============================================================
# EMAIL FORM
# ============================================================

def render_email_form():

    if "travel_info" not in st.session_state:
        return

    st.divider()

    st.subheader("📧 Send Travel Information")

    send_email_option = st.radio(
        "Do you want to send this information via email?",
        ["No", "Yes"],
        horizontal=True,
    )

    if send_email_option != "Yes":
        return

    with st.form("email_form"):

        sender_email = st.text_input(
            "Your Email",
            placeholder="your@email.com",
        )

        receiver_email = st.text_input(
            "Receiver Email",
            placeholder="receiver@email.com",
        )

        subject = st.text_input(
            "Email Subject",
            value="AI Travel Information",
        )

        submit_button = st.form_submit_button(
            "📨 Send Email"
        )

    if submit_button:

        if not receiver_email.strip():

            st.warning(
                "Please enter the receiver's email address."
            )

            return

        try:

            send_email(
                sender_email=sender_email.strip(),
                receiver_email=receiver_email.strip(),
                subject=subject.strip(),
                travel_information=(
                    st.session_state.travel_info
                ),
            )

            st.success(
                "✅ Travel information sent successfully!"
            )

        except Exception as exc:

            st.error(
                f"❌ Unable to send email: {exc}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    render_custom_css()

    render_sidebar()

    render_header()

    user_input = render_travel_input()

    if st.button(
        "✈️ Get Travel Information",
        type="primary",
        use_container_width=True,
    ):

        process_query(user_input)

    render_travel_information()

    render_email_form()

    st.markdown(
        """
        <div class="footer">
        AI Travel Agent • Streamlit • LangChain • Google Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
```
