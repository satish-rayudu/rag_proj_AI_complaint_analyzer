import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Complaint Analyzer",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    opacity: 0.75;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">🤖 AI Complaint Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Domain-Independent RAG-Based Complaint Analysis System</div>',
    unsafe_allow_html=True
)

st.write(
    "Analyze customer complaints using semantic retrieval "
    "and domain-specific complaint policies."
)


# =========================================================
# KNOWLEDGE BASE
# =========================================================

domain_documents = [

"""DOMAIN: E-Commerce
CATEGORY: Payment Failed

If a customer reports that a payment failed but money was deducted,
the transaction should be verified. If the transaction remains failed,
the amount should be refunded according to the applicable refund policy.
""",

"""DOMAIN: E-Commerce
CATEGORY: Wrong Product

If a customer receives a product different from the ordered product,
the issue should be classified as a wrong-product complaint.
Replacement or refund may be provided after order verification.
""",

"""DOMAIN: E-Commerce
CATEGORY: Damaged Product

If a product arrives damaged, the complaint should be classified as
product damage. The customer may be asked to provide order details
and photographs. Replacement or refund may be provided after verification.
""",

"""DOMAIN: E-Commerce
CATEGORY: Late Delivery

If an order has not arrived by the expected delivery date,
the delivery status and tracking information should be checked.
""",

"""DOMAIN: Banking
CATEGORY: Failed Transaction

If a banking transaction fails but the customer's account is debited,
the transaction should be verified and the amount should be reconciled
or refunded according to the bank's process.
""",

"""DOMAIN: Banking
CATEGORY: Unauthorized Transaction

Reports of unauthorized transactions should be treated as high-priority
financial complaints and may require immediate investigation and escalation.
""",

"""DOMAIN: Banking
CATEGORY: Card Issue

Problems involving blocked, damaged, or malfunctioning cards should be
verified using the customer's account and card information.
""",

"""DOMAIN: Telecom
CATEGORY: Network Problem

Complaints about poor or unavailable network service should be classified
as network-related complaints. Service availability and outage information
should be checked.
""",

"""DOMAIN: Telecom
CATEGORY: Billing Problem

Unexpected or incorrect charges should be classified as billing complaints.
The customer's bill and usage records should be reviewed.
""",

"""DOMAIN: Healthcare
CATEGORY: Appointment Complaint

Issues involving appointment delays, cancellations, or scheduling should
be classified as appointment-related complaints.
""",

"""DOMAIN: Travel
CATEGORY: Flight Cancellation

If a customer's flight is cancelled, booking status and applicable
refund or rebooking options should be checked.
""",

"""DOMAIN: Travel
CATEGORY: Booking Problem

Problems involving incorrect or failed bookings should be verified
using booking information.
""",

"""DOMAIN: IT / SaaS
CATEGORY: Account Access

If a customer cannot access an account, the issue should be classified
as an account-access complaint. Password recovery or account recovery
procedures should be followed.
""",

"""DOMAIN: IT / SaaS
CATEGORY: Technical Problem

Software errors or service failures should be classified as technical
complaints and investigated using available system information.
"""
]


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# FAISS VECTOR DATABASE
# =========================================================

@st.cache_resource
def create_vectorstore():

    return FAISS.from_texts(
        domain_documents,
        embedding_model
    )


domain_vectorstore = create_vectorstore()


# =========================================================
# RETRIEVER
# =========================================================

domain_retriever = domain_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)


# =========================================================
# COMPLAINT ANALYZER
# =========================================================

def analyze_complaint(complaint, retrieved_context):

    text = complaint.lower()


    # =====================================================
    # DOMAIN DETECTION
    # =====================================================

    if any(x in text for x in [
        "flight",
        "hotel",
        "travel",
        "airport",
        "airline",
        "rebooking"
    ]):
        domain = "Travel"

    elif any(x in text for x in [
        "hospital",
        "doctor",
        "appointment",
        "medical",
        "healthcare",
        "clinic"
    ]):
        domain = "Healthcare"

    elif any(x in text for x in [
        "network",
        "internet",
        "sim",
        "mobile",
        "recharge",
        "signal",
        "telecom"
    ]):
        domain = "Telecom"

    elif any(x in text for x in [
        "bank",
        "banking",
        "transaction",
        "account debited",
        "account was charged",
        "money deducted",
        "money was deducted",
        "card",
        "debit card",
        "credit card"
    ]):
        domain = "Banking"

    elif any(x in text for x in [
        "software",
        "login",
        "password",
        "subscription",
        "application",
        "app",
        "account access",
        "technical"
    ]):
        domain = "IT / SaaS"

    elif any(x in text for x in [
        "product",
        "order",
        "package",
        "delivery",
        "smartphone",
        "laptop",
        "item",
        "refund",
        "online"
    ]):
        domain = "E-Commerce"

    else:
        domain = "General"


    # =====================================================
    # CATEGORY DETECTION
    # =====================================================

    # ---------------- E-COMMERCE ----------------

    if any(x in text for x in [
        "wrong product",
        "different product",
        "wrong item",
        "different item",
        "received the wrong",
        "received a different",
        "package contained a different",
        "sent the wrong"
    ]):
        category = "Wrong Product"


    elif any(x in text for x in [
        "damaged product",
        "product damaged",
        "broken product",
        "arrived damaged",
        "package damaged",
        "item damaged"
    ]):
        category = "Damaged Product"


    elif any(x in text for x in [
        "late delivery",
        "delivery delayed",
        "order not arrived",
        "not delivered",
        "order hasn't arrived",
        "order has not arrived"
    ]):
        category = "Late Delivery"


    # ---------------- BANKING ----------------

    elif any(x in text for x in [
        "transaction failed",
        "failed transaction",
        "payment failed",
        "money deducted",
        "money was deducted",
        "account debited",
        "account was charged",
        "transaction that failed",
        "charged for a transaction that failed"
    ]):
        category = "Failed Transaction"


    elif any(x in text for x in [
        "unauthorized transaction",
        "unknown transaction",
        "did not make this transaction",
        "someone used my card",
        "fraudulent transaction",
        "fraud transaction"
    ]):
        category = "Unauthorized Transaction"


    # FIXED CARD ISSUE SECTION

    elif any(x in text for x in [
        "card blocked",
        "card not working",
        "card has stopped working",
        "card stopped working",
        "debit card not working",
        "debit card has stopped working",
        "debit card stopped working",
        "credit card not working",
        "credit card has stopped working",
        "credit card stopped working",
        "damaged card",
        "card malfunction",
        "card is not working",
        "card isn't working",
        "cannot use my card",
        "can't use my card",
        "unable to use my card"
    ]):
        category = "Card Issue"


    # ---------------- TELECOM ----------------

    elif any(x in text for x in [
        "network problem",
        "network issue",
        "network keeps disconnecting",
        "network is disconnecting",
        "network disconnecting",
        "no network",
        "poor network",
        "no signal",
        "internet not working",
        "internet has not been working",
        "internet keeps disconnecting",
        "internet has been disconnecting",
        "internet is disconnecting",
        "internet disconnecting",
        "mobile internet",
        "network unavailable",
        "connection keeps dropping",
        "internet keeps dropping"
    ]):
        category = "Network Problem"


    elif any(x in text for x in [
        "incorrect bill",
        "wrong bill",
        "billing problem",
        "billing issue",
        "unexpected charge",
        "incorrect charge",
        "charged extra",
        "wrong charge"
    ]):
        category = "Billing Problem"


    # ---------------- HEALTHCARE ----------------

    elif any(x in text for x in [
        "appointment delayed",
        "appointment cancelled",
        "appointment canceled",
        "appointment problem",
        "appointment issue",
        "scheduling problem",
        "appointment was cancelled",
        "appointment was canceled",
        "reschedule my appointment",
        "reschedule it",
        "appointment was delayed",
        "doctor appointment"
    ]):
        category = "Appointment Complaint"


    elif any(x in text for x in [
        "medical bill",
        "hospital bill",
        "healthcare bill",
        "medical charge",
        "unexpected medical charge",
        "billing at hospital"
    ]):
        category = "Billing Complaint"


    elif any(x in text for x in [
        "poor service",
        "bad service",
        "hospital service",
        "doctor service",
        "staff behavior"
    ]):
        category = "Service Complaint"


    # ---------------- TRAVEL ----------------

    elif any(x in text for x in [
        "flight cancelled",
        "flight canceled",
        "cancelled flight",
        "canceled flight",
        "flight was cancelled",
        "flight was canceled"
    ]):
        category = "Flight Cancellation"


    elif any(x in text for x in [
        "booking failed",
        "booking problem",
        "incorrect booking",
        "wrong booking",
        "booking issue"
    ]):
        category = "Booking Problem"


    elif any(x in text for x in [
        "flight delayed",
        "travel delayed",
        "travel delay",
        "flight delay"
    ]):
        category = "Travel Delay"


    # ---------------- IT / SAAS ----------------

    elif any(x in text for x in [
        "cannot login",
        "can't login",
        "unable to login",
        "cannot access my account",
        "can't access my account",
        "unable to access my account",
        "forgot password",
        "password reset",
        "account access"
    ]):
        category = "Account Access"


    elif any(x in text for x in [
        "software error",
        "application error",
        "app not working",
        "software not working",
        "service failure",
        "technical issue",
        "technical problem",
        "system error"
    ]):
        category = "Technical Problem"


    elif any(x in text for x in [
        "subscription problem",
        "subscription issue",
        "subscription charge",
        "subscription cancelled",
        "renewal problem",
        "renewal issue"
    ]):
        category = "Subscription Problem"


    else:
        category = "General Complaint"


    # =====================================================
    # SEVERITY & PRIORITY
    # =====================================================

    if any(x in text for x in [
        "unauthorized",
        "fraud",
        "security",
        "financial loss",
        "stolen"
    ]):
        severity = "High"
        priority = "High"


    elif any(x in text for x in [
        "money deducted",
        "money was deducted",
        "account debited",
        "account was charged",
        "urgent",
        "as soon as possible",
        "unable to",
        "cannot access",
        "not working",

        # Card-related priority
        "card not working",
        "card has stopped working",
        "card stopped working",
        "debit card not working",
        "debit card has stopped working",
        "debit card stopped working",
        "credit card not working",
        "credit card has stopped working",
        "credit card stopped working",
        "cannot use my card",
        "can't use my card",
        "unable to use my card"
    ]):
        severity = "Medium"
        priority = "High"


    else:
        severity = "Medium"
        priority = "Medium"


    # =====================================================
    # RECOMMENDED ACTION
    # =====================================================

    actions = {

        "Wrong Product":
            "Verify the order details and provide a replacement or refund according to policy.",

        "Damaged Product":
            "Verify the damage and provide a replacement or refund according to policy.",

        "Late Delivery":
            "Check the order tracking and delivery status.",

        "Failed Transaction":
            "Verify the transaction and reconcile or refund the deducted amount.",

        "Unauthorized Transaction":
            "Immediately investigate the transaction and escalate the complaint.",

        "Card Issue":
            "Verify the customer's card status and provide appropriate card support.",

        "Network Problem":
            "Check service availability and network outage information and provide technical assistance.",

        "Billing Problem":
            "Review the customer's bill and usage records.",

        "Appointment Complaint":
            "Verify the appointment details and provide appropriate scheduling assistance.",

        "Billing Complaint":
            "Review the healthcare billing records and verify the applicable charges.",

        "Service Complaint":
            "Review the service complaint and escalate it when customer impact is significant.",

        "Flight Cancellation":
            "Check the booking status and provide applicable refund or rebooking options.",

        "Booking Problem":
            "Verify the booking information and provide an appropriate resolution.",

        "Travel Delay":
            "Check the current travel or booking status and provide appropriate assistance.",

        "Account Access":
            "Guide the customer through password reset or account recovery procedures.",

        "Technical Problem":
            "Investigate the technical issue using available system information.",

        "Subscription Problem":
            "Review the customer's subscription details and resolve the subscription issue.",

        "General Complaint":
            "Review the complaint and request additional information if necessary."
    }


    action = actions[category]


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "Domain": domain,

        "Complaint Category": category,

        "Complaint Summary":
            complaint.strip().replace("\n", " "),

        "Issue Identified": category,

        "Severity": severity,

        "Priority": priority,

        "Relevant Policy": retrieved_context,

        "Recommended Action": action
    }


# =========================================================
# STREAMLIT INTERFACE
# =========================================================

st.markdown("---")

st.header("📝 Enter Customer Complaint")

complaint = st.text_area(
    "Complaint",
    placeholder=(
        "Example: My internet has been disconnecting "
        "frequently for the past three days."
    ),
    height=150
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze Complaint", type="primary"):

    if not complaint.strip():

        st.warning("Please enter a complaint.")

    else:

        with st.spinner("Analyzing complaint..."):

            # ---------------------------------------------
            # RAG RETRIEVAL
            # ---------------------------------------------

            retrieved_docs = domain_retriever.invoke(
                complaint
            )

            retrieved_context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            # ---------------------------------------------
            # COMPLAINT ANALYSIS
            # ---------------------------------------------

            result = analyze_complaint(
                complaint,
                retrieved_context
            )


        st.success(
            "Complaint analysis completed!"
        )

        st.markdown("---")

        st.header(
            "📊 AI Complaint Analysis"
        )


        # =================================================
        # DOMAIN + CATEGORY
        # =================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Domain",
                result["Domain"]
            )

        with col2:

            st.metric(
                "Complaint Category",
                result["Complaint Category"]
            )


        # =================================================
        # SEVERITY + PRIORITY
        # =================================================

        col3, col4 = st.columns(2)

        with col3:

            st.metric(
                "Severity",
                result["Severity"]
            )

        with col4:

            st.metric(
                "Priority",
                result["Priority"]
            )


        # =================================================
        # COMPLAINT SUMMARY
        # =================================================

        st.subheader(
            "📌 Complaint Summary"
        )

        st.write(
            result["Complaint Summary"]
        )


        # =================================================
        # ISSUE
        # =================================================

        st.subheader(
            "🔎 Issue Identified"
        )

        st.info(
            result["Issue Identified"]
        )


        # =================================================
        # RECOMMENDED ACTION
        # =================================================

        st.subheader(
            "💡 Recommended Action"
        )

        st.success(
            result["Recommended Action"]
        )


        # =================================================
        # RETRIEVED POLICY
        # =================================================

        st.subheader(
            "📚 Retrieved Policy"
        )

        with st.expander(
            "View relevant knowledge retrieved by RAG"
        ):

            st.write(
                result["Relevant Policy"]
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "📚 Knowledge Base"
    )

    st.write(
        "Supported Domains"
    )

    st.write("🛒 E-Commerce")
    st.write("🏦 Banking")
    st.write("📡 Telecom")
    st.write("🏥 Healthcare")
    st.write("✈️ Travel")
    st.write("💻 IT / SaaS")


    st.markdown("---")


    st.header(
        "⚙️ RAG System"
    )

    st.write(
        "🔹 Sentence Transformers"
    )

    st.write(
        "🔹 FAISS Vector Database"
    )

    st.write(
        "🔹 Semantic Retrieval"
    )

    st.write(
        "🔹 Complaint Classification"
    )


    st.markdown("---")


    st.info(
        "The system retrieves relevant domain "
        "knowledge from the FAISS vector database "
        "before analyzing the complaint."
    )


    st.markdown("---")


    st.caption(
        "AI Complaint Analyzer"
    )

    st.caption(
        "RAG Project"
    )