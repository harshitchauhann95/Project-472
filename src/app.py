import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Smart Hostel Complaint & Maintenance System",
    page_icon="🛠️",
    layout="wide"
)

# Load Models and Base Dataset
@st.cache_resource
def load_system_artifacts():
    m_path = 'models/complaint_classifier.pkl' if os.path.exists('models/complaint_classifier.pkl') else '../models/complaint_classifier.pkl'
    u_path = 'models/urgency_classifier.pkl' if os.path.exists('models/urgency_classifier.pkl') else '../models/urgency_classifier.pkl'
    t_path = 'models/tfidf_vectorizer.pkl' if os.path.exists('models/tfidf_vectorizer.pkl') else '../models/tfidf_vectorizer.pkl'
    d_path = 'data/smart_hostel_complaints_dataset.csv' if os.path.exists('data/smart_hostel_complaints_dataset.csv') else '../data/smart_hostel_complaints_dataset.csv'
    
    clf = joblib.load(m_path)
    urgency_clf = joblib.load(u_path)
    tfidf = joblib.load(t_path)
    df = pd.read_csv(d_path)
    return clf, urgency_clf, tfidf, df

clf, urgency_clf, tfidf, df = load_system_artifacts()

# Session State for Live Ticket Tracking
if 'tickets' not in st.session_state:
    st.session_state['tickets'] = [
        {
            "ticket_id": f"TCK-{1001 + i}",
            "block": row['hostel_block'],
            "room": str(row['room_number']),
            "description": row['description'],
            "category": row['category'],
            "urgency": row['urgency'],
            "assigned_team": row['assigned_team'],
            "status": row['status'] if pd.notna(row['status']) else "Reported",
            "created_at": row['created_at']
        }
        for i, row in df.head(10).iterrows()
    ]

TEAM_ROUTING_MAP = {
    'Plumbing': 'Plumbing & Sanitation Squad',
    'Electrical': 'Electrical Maintenance Team',
    'Furniture': 'Carpentry & Furniture Unit',
    'Civil': 'Civil & Masonry Crew',
    'Network': 'IT & Network Support Desk',
    'Housekeeping': 'Hostel Housekeeping Team'
}

# Sidebar Navigation Role Selector
st.sidebar.title("Navigation & Roles")
role = st.sidebar.selectbox("Select Portal", ["Resident Portal", "Maintenance Staff Portal", "Warden Analytics Dashboard"])

st.sidebar.divider()
st.sidebar.info("Project ID: CSEAIML2026-P39\nSmart Hostel Tracking System")

# ==========================================
# 1. RESIDENT PORTAL
# ==========================================
if role == "Resident Portal":
    st.title("🛠️ Resident Complaint Portal")
    st.markdown("Submit maintenance issues, attach photo references, and track your ticket status in real-time.")
    
    tab1, tab2 = st.tabs(["Submit New Complaint", "Track My Tickets"])
    
    with tab1:
        with st.form("resident_complaint_form"):
            col1, col2 = st.columns(2)
            with col1:
                hostel_block = st.selectbox("Hostel Block", ["Block A", "Block B", "Block C", "Block D", "Block E"])
            with col2:
                room_number = st.text_input("Room Number", placeholder="e.g., 204")
                
            description = st.text_area("Detailed Description of Issue", placeholder="Describe what's broken (e.g., ceiling fan sparking, leaking tap)...")
            uploaded_photo = st.file_uploader("Upload Issue Photo (Optional)", type=["jpg", "png", "jpeg"])
            
            submitted = st.form_submit_button("Submit Complaint")
            
        if submitted:
            if not description.strip() or not room_number.strip():
                st.warning("Please fill in both the room number and complaint description.")
            else:
                # AI Prediction
                vec = tfidf.transform([description.lower()])
                pred_category = clf.predict(vec)[0]
                pred_urgency = urgency_clf.predict(vec)[0]
                assigned_team = TEAM_ROUTING_MAP.get(pred_category, "General Maintenance")
                
                new_ticket = {
                    "ticket_id": f"TCK-{1000 + len(st.session_state['tickets']) + 1}",
                    "block": hostel_block,
                    "room": room_number,
                    "description": description,
                    "category": pred_category,
                    "urgency": pred_urgency,
                    "assigned_team": assigned_team,
                    "status": "Reported",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                st.session_state['tickets'].insert(0, new_ticket)
                st.success("Complaint successfully submitted and routed!")
                st.balloons()
                
                st.info(f"📬 **Notification Sent:** Your ticket (`{new_ticket['ticket_id']}`) has been automatically routed to the **{assigned_team}** with **{pred_urgency}** priority.")

    with tab2:
        st.subheader("Live Ticket Status Tracker")
        search_room = st.text_input("Filter by Room Number", placeholder="Enter room number...")
        
        tickets_df = pd.DataFrame(st.session_state['tickets'])
        if search_room:
            tickets_df = tickets_df[tickets_df['room'].str.contains(search_room, case=False, na=False)]
            
        for _, t in tickets_df.iterrows():
            with st.expander(f"[{t['ticket_id']}] Room {t['room']} ({t['category']}) - Status: {t['status']}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Block:** {t['block']}")
                    st.write(f"**Assigned Team:** {t['assigned_team']}")
                    st.write(f"**Urgency:** {t['urgency']}")
                with col_b:
                    st.write(f"**Status:** `{t['status']}`")
                    st.write(f"**Created At:** {t['created_at']}")
                st.markdown(f"**Description:** {t['description']}")

# ==========================================
# 2. MAINTENANCE STAFF PORTAL
# ==========================================
elif role == "Maintenance Staff Portal":
    st.title("🔧 Maintenance Staff Workdesk")
    st.markdown("View routed tickets assigned to your unit and update resolution progress.")
    
    selected_team = st.selectbox("Select Your Maintenance Unit", list(TEAM_ROUTING_MAP.values()))
    
    tickets_df = pd.DataFrame(st.session_state['tickets'])
    team_tickets = tickets_df[tickets_df['assigned_team'] == selected_team] if not tickets_df.empty else pd.DataFrame()
    
    if team_tickets.empty:
        st.info(f"No active tickets assigned to {selected_team}.")
    else:
        st.subheader(f"Assigned Tasks for {selected_team}")
        for idx, t in team_tickets.iterrows():
            with st.container():
                cols = st.columns([3, 1, 1, 1])
                cols[0].write(f"**[{t['ticket_id']}] Room {t['room']} ({t['block']})**\n\n{t['description']}")
                cols[1].write(f"Urgency: **{t['urgency']}**")
                cols[2].write(f"Status: **{t['status']}**")
                
                new_status = cols[3].selectbox("Update", ["Reported", "In Progress", "Resolved"], index=["Reported", "In Progress", "Resolved"].index(t['status']), key=f"status_{t['ticket_id']}")
                
                if new_status != t['status']:
                    st.session_state['tickets'][idx]['status'] = new_status
                    st.success(f"Ticket {t['ticket_id']} updated to {new_status}!")
                    st.rerun()
                st.divider()

# ==========================================
# 3. WARDEN ANALYTICS DASHBOARD
# ==========================================
elif role == "Warden Analytics Dashboard":
    st.title("📊 Warden Analytics & Operations Dashboard")
    st.markdown("Monitor hostel-wide maintenance metrics, open complaints, resolution times, and recurring issue patterns.")
    
    tickets_df = pd.DataFrame(st.session_state['tickets'])
    
    col1, col2, col3, col4 = st.columns(4)
    total_tickets = len(tickets_df)
    open_tickets = len(tickets_df[tickets_df['status'] != 'Resolved']) if not tickets_df.empty else 0
    resolved_tickets = total_tickets - open_tickets
    avg_res_time = df['resolution_time_hours'].mean() if 'resolution_time_hours' in df.columns else 18.5
    
    col1.metric("Total Complaints", total_tickets)
    col2.metric("Open / Pending", open_tickets, delta="-2" if open_tickets > 0 else "0")
    col3.metric("Resolved Tickets", resolved_tickets)
    col4.metric("Avg Resolution Time", f"{avg_res_time:.1f} hrs")
    
    st.divider()
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Complaints by Hostel Block (Hotspot Analysis)")
        if not tickets_df.empty and 'block' in tickets_df.columns:
            block_counts = tickets_df['block'].value_counts()
            st.bar_chart(block_counts)
        else:
            st.info("No block data available.")
            
    with col_chart2:
        st.subheader("Complaints by Category")
        if not tickets_df.empty and 'category' in tickets_df.columns:
            cat_counts = tickets_df['category'].value_counts()
            st.bar_chart(cat_counts)
        else:
            st.info("No category data available.")
            
    st.subheader("Full System Ticket Audit Log")
    st.dataframe(tickets_df, use_container_width=True)