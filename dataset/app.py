
import streamlit as st
import joblib

# Load the trained model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)
# Message statistics
if "total" not in st.session_state:
    st.session_state["total"] = 0

if "spam" not in st.session_state:
    st.session_state["spam"] = 0

if "ham" not in st.session_state:
    st.session_state["ham"] = 0

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #121212;
    color: #FFFFFF;
}

[data-testid="stHeader"] {
    background-color: #121212;
}

.title {
    text-align: center;
    color: #64B5F6;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #E0E0E0;
    font-size: 16px;
}

/* Make text readable */
.stApp p,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stMarkdown {
    color: #FFFFFF;
}

/* Message input box */
.stTextArea textarea {
    background-color: #252525;
    color: #FFFFFF;
    border: 1px solid #64B5F6;
}

/* Buttons */
.stButton button {
    background-color: #1976D2;
    color: #FFFFFF;
    border-radius: 8px;
    border: none;
}

/* Info box */
[data-testid="stAlert"] {
    background-color: #252525;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">📧 Email Spam Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered message classification</div>',
    unsafe_allow_html=True
)

st.write("")
st.write("### 🔍 Check Your Message")

st.info(
    "Enter an email or SMS message below to check "
    "whether it looks like spam."
)

# Message input
message = st.text_area(
    "Your message",
    placeholder="Example: Congratulations! You won a free prize...",
    height=150
)

# Buttons
col1, col2 = st.columns(2)

with col1:
    check = st.button("🔎 Check Message", use_container_width=True)

with col2:
    clear = st.button("🗑️ Clear Message", use_container_width=True)

if clear:
    st.rerun()

if check:
    if message.strip():
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]
        st.session_state.total += 1

        if prediction == 1:
            st.session_state.spam += 1
        else:
            st.session_state.ham += 1

        st.write("---")
        st.subheader("📊 Analysis Result")

        if prediction == 1:
            st.error("⚠️ This message is SPAM!")
            st.write(
                "Be careful. Avoid clicking suspicious links "
                "or sharing personal information."
            )
        else:
            st.success("✅ This message is NOT SPAM!")
            st.write(
                "The model classified this as a normal message. "
                "Still, be careful with unknown senders and links."
            )

        st.caption(
            "Note: Predictions are not guaranteed to be correct."
        )
    else:
        st.warning("Please enter a message first.")
st.write("---")
st.subheader("📊 Message Dashboard")

c1, c2, c3 = st.columns(3)

c1.metric("Total Checked", st.session_state.total)
c2.metric("Spam", st.session_state.spam)
c3.metric("Not Spam", st.session_state.ham)

if st.session_state.total > 0:
    st.bar_chart({
        "Messages": {
            "Spam": st.session_state.spam,
            "Not Spam": st.session_state.ham
        }
    })

st.write("---")
st.caption("Developed using Python, Machine Learning and Streamlit")
