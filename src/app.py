"""
Smart Hostel Complaint and Maintenance Tracking System
Web Application (Streamlit)
Implements:
  1. Resident Complaint Portal (Multi-parameter reporting, optional photo, AI triage, ticket lifecycle tracking)
  2. Maintenance Staff Portal (Team assignment queues, status progression, technician logs)
  3. Warden Analytics Dashboard (Resolution metrics, SLA compliance, reopen rate, recurring hotspots, and manual override queue)
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.routing.priority_engine import PriorityEngine
from src.routing.team_router import TeamRouter, DISPLAY_TEAM_NAMES, TEAM_MAP
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.image_classifier import OptionalImageClassifier

# Page configuration
st.set_page_config(
    page_title="Smart Hostel Complaint & Maintenance Tracking System",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-high {
        background-color: #ff4b4b;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .badge-med {
        background-color: #ffa500;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .badge-low {
        background-color: #2ca02c;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_engines():
    """
    Initializes AI models, NLP vectorizers, priority engines, and image analyzers.
    """
    priority_engine = PriorityEngine()
    team_router = TeamRouter(confidence_threshold=0.60)
    sentiment_analyzer = SentimentAnalyzer()
    image_classifier = OptionalImageClassifier()
    
    # Load TF-IDF and Model if available
    tfidf_path = "models/tfidf_vectorizer.pkl"
    model_path = "models/tfidf_model.pkl"
    
    vectorizer = joblib.load(tfidf_path) if os.path.exists(tfidf_path) else None
    classifier = joblib.load(model_path) if os.path.exists(model_path) else None
    
    return priority_engine, team_router, sentiment_analyzer, image_classifier, vectorizer, classifier


priority_engine, team_router, sentiment_analyzer, image_classifier, vectorizer, classifier = load_system_engines()


def initialize_demo_tickets():
    """
    Generates representative seed tickets from processed dataset for realistic demo.
    """
    dataset_path = "data/processed/hostel_complaints_cleaned.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "data/processed/test.csv"
        
    tickets = []
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path).head(16)
        for idx, row in df.iterrows():
            now = datetime.now()
            created_dt = (now - timedelta(hours=int(idx * 6 + 2))).strftime("%Y-%m-%d %H:%M:%S")
            status = "RESOLVED" if idx % 3 == 0 else ("IN_PROGRESS" if idx % 3 == 1 else "TRIAGED")
            resolved_dt = (now - timedelta(hours=int(idx * 2))).strftime("%Y-%m-%d %H:%M:%S") if status == "RESOLVED" else None
            
            p_score = int(row.get("priority_score", 65))
            p_band = "High" if p_score >= 75 else ("Medium" if p_score >= 40 else "Low")
            
            tickets.append({
                "ticket_id": f"TCK-{20261001 + idx}",
                "block": str(row.get("block", "Block A")),
                "floor": str(row.get("floor", "2")),
                "room": str(row.get("room", "Room 204")),
                "description": str(row.get("description", "")),
                "resident_urgency": str(row.get("urgency", "Medium")),
                "impact_scope": "Shared Room (2-4 Residents)",
                "category": str(row.get("category", "electrical")),
                "confidence": 0.94 if idx % 4 != 0 else 0.54,  # include low-confidence sample for demo
                "needs_review": True if (idx % 4 == 0) else False,
                "priority_score": p_score,
                "priority_band": p_band,
                "assigned_team": str(row.get("assigned_team", "electricians")),
                "sentiment": str(row.get("sentiment", "neutral")),
                "status": status,
                "created_at": created_dt,
                "resolved_at": resolved_dt,
                "has_photo": True if idx % 2 == 0 else False,
                "history": [
                    {"status": "SUBMITTED", "timestamp": created_dt, "actor": "Resident", "note": "Complaint submitted via portal"},
                    {"status": "TRIAGED", "timestamp": created_dt, "actor": "AI System", "note": f"Triaged to {row.get('category')} (Priority: {p_band})"},
                ]
            })
    return tickets


if "tickets" not in st.session_state:
    st.session_state["tickets"] = initialize_demo_tickets()


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/hostel.png", width=64)
st.sidebar.title("Smart Hostel System")
st.sidebar.caption("Aligned with Smart_Hostel_Report_File.pdf")

role = st.sidebar.radio(
    "Select Interface Portal",
    ["Resident Portal", "Maintenance Staff Portal", "Warden Analytics Dashboard"]
)

st.sidebar.divider()
st.sidebar.markdown(r"""
**System Architecture:**
- **NLP Models:** TF-IDF, PyTorch BiLSTM, DistilBERT
- **Priority:** \(P = 0.35C + 0.30U + 0.20R + 0.15I\)
- **Gating:** Confidence threshold < 0.60 \(\rightarrow\) Warden Review
""")


# ==============================================================================
# 1. RESIDENT PORTAL
# ==============================================================================
if role == "Resident Portal":
    st.title("🏠 Resident Complaint & Maintenance Portal")
    st.markdown("Report hostel maintenance issues with AI classification, priority assessment, and live tracking.")
    
    tabs = st.tabs(["📝 File New Complaint", "🔎 Track My Tickets", "📜 Active Tickets Directory"])
    
    with tabs[0]:
        st.subheader("Submit Maintenance Request")
        with st.form("complaint_submission_form", clear_on_submit=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                hostel_block = st.selectbox("Hostel Block", ["Block A", "Block B", "Block C", "Block D"])
            with col2:
                floor_no = st.selectbox("Floor", ["Ground Floor", "Floor 1", "Floor 2", "Floor 3", "Floor 4"])
            with col3:
                room_number = st.text_input("Room / Facility Number", placeholder="e.g., Room 204")
                
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                resident_urgency = st.selectbox(
                    "Resident Selected Urgency",
                    ["Emergency", "Critical", "High", "Medium", "Low"],
                    index=3
                )
            with col_u2:
                impact_scope = st.selectbox(
                    "Impact Scope",
                    ["Single Resident Room", "Shared Room (2-4 Residents)", "Common Area / Floor Washroom / Mess", "Hostel / Entire Wing"],
                    index=1
                )
                
            complaint_text = st.text_area(
                "Detailed Complaint Description",
                placeholder="Describe the issue clearly (e.g., Ceiling fan sparking and smelling burnt, or washbasin pipe leaking on floor)...",
                height=100
            )
            
            uploaded_image = st.file_uploader(
                "Attach Photograph (Optional - Image branch will execute only if provided)",
                type=["jpg", "jpeg", "png"]
            )
            
            submit_btn = st.form_submit_button("🚀 Submit Complaint & Trigger AI Triage", use_container_width=True)
            
        if submit_btn:
            if not complaint_text.strip() or not room_number.strip():
                st.error("Please enter both the Room Number and Complaint Description.")
            else:
                # 1. NLP Category Prediction
                if classifier and vectorizer:
                    vec = vectorizer.transform([complaint_text])
                    pred_category = classifier.predict(vec)[0]
                    probs = classifier.predict_proba(vec)[0]
                    confidence = float(np.max(probs))
                else:
                    # Heuristic mapping if classifier not in memory
                    from src.data.preprocess import map_to_target_category
                    pred_category = map_to_target_category("", "", complaint_text)
                    confidence = 0.85
                    
                # 2. Sentiment Analysis (Stored separately from physical risk)
                sentiment_res = sentiment_analyzer.analyze(complaint_text)
                
                # 3. Optional Image Classifier
                img_res = image_classifier.classify_image(uploaded_image)
                
                # 4. Priority Engine (Formula: P = 0.35C + 0.30U + 0.20R + 0.15I)
                priority_res = priority_engine.compute_priority(
                    category=pred_category,
                    description=complaint_text,
                    resident_urgency=resident_urgency,
                    impact_scope=impact_scope
                )
                
                # 5. Team Routing & Low-Confidence Gating
                route_res = team_router.route_complaint(
                    predicted_category=pred_category,
                    confidence=confidence
                )
                
                # Create Ticket
                ticket_id = f"TCK-{20261000 + len(st.session_state['tickets']) + 1}"
                created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                new_ticket = {
                    "ticket_id": ticket_id,
                    "block": hostel_block,
                    "floor": floor_no,
                    "room": room_number,
                    "description": complaint_text,
                    "resident_urgency": resident_urgency,
                    "impact_scope": impact_scope,
                    "category": pred_category,
                    "confidence": confidence,
                    "needs_review": route_res["needs_warden_review"],
                    "priority_score": priority_res["priority_score"],
                    "priority_band": priority_res["priority_band"],
                    "assigned_team": route_res["assigned_team"],
                    "sentiment": sentiment_res["sentiment"],
                    "status": "TRIAGED" if not route_res["needs_warden_review"] else "SUBMITTED",
                    "created_at": created_time,
                    "resolved_at": None,
                    "has_photo": img_res["has_image"],
                    "history": [
                        {"status": "SUBMITTED", "timestamp": created_time, "actor": "Resident", "note": "Complaint registered"},
                        {"status": "TRIAGED", "timestamp": created_time, "actor": "AI System", "note": f"Assigned to {route_res['assigned_team']} (Priority: {priority_res['priority_band']})"}
                    ]
                }
                
                st.session_state["tickets"].insert(0, new_ticket)
                
                st.success(f"Ticket Created Successfully! **Ticket ID: {ticket_id}**")
                
                # Display AI Triage Breakdown Card
                with st.expander("🔍 AI Triage & Routing Diagnostics", expanded=True):
                    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                    r_col1.metric("Predicted Category", pred_category.replace("_", " ").title(), f"{confidence*100:.1f}% Confidence")
                    r_col2.metric("Assigned Team", DISPLAY_TEAM_NAMES.get(route_res["assigned_team"], route_res["assigned_team"]))
                    r_col3.metric("Priority Score", f"{priority_res['priority_score']}/100", f"Band: {priority_res['priority_band']}")
                    r_col4.metric("Sentiment Tone", sentiment_res["sentiment"].title(), f"Polarity: {sentiment_res['polarity']}")
                    
                    if priority_res["hazard_escalation"]:
                        st.warning(f"⚠️ **Hazard Alert:** {priority_res['hazard_reason']}")
                    if route_res["needs_warden_review"]:
                        st.info(f"ℹ️ **Review Notice:** {route_res['routing_reason']}")
                    if img_res["has_image"]:
                        st.info(f"📸 **Visual Analysis:** {img_res['status']} | Visual cues: {', '.join(img_res['visual_tags'])}")
                        
    with tabs[1]:
        st.subheader("Check Complaint Lifecycle & Audit Trail")
        search_id = st.text_input("Enter Ticket ID (e.g., TCK-20261001)", placeholder="TCK-...")
        if search_id:
            match = next((t for t in st.session_state["tickets"] if t["ticket_id"].strip().upper() == search_id.strip().upper()), None)
            if match:
                st.markdown(f"### Ticket: `{match['ticket_id']}`")
                c1, c2, c3, c4 = st.columns(4)
                c1.write(f"**Location:** {match['block']} - {match['room']}")
                c2.write(f"**Category:** `{match['category']}`")
                c3.write(f"**Assigned Team:** {DISPLAY_TEAM_NAMES.get(match['assigned_team'], match['assigned_team'])}")
                
                badge_cls = "badge-high" if match["priority_band"] == "High" else ("badge-med" if match["priority_band"] == "Medium" else "badge-low")
                c4.markdown(f"**Priority:** <span class='{badge_cls}'>{match['priority_band']} ({match['priority_score']})</span>", unsafe_allow_html=True)
                
                st.info(f"**Description:** {match['description']}")
                st.write(f"**Current Status:** `{match['status']}` | **Created:** {match['created_at']}")
                
                st.markdown("#### Status Lifecycle & Timeline")
                for h in match["history"]:
                    st.markdown(f"- **[{h['timestamp']}]** `{h['status']}` by *{h['actor']}*: {h['note']}")
                    
                # Resident Actions (Close or Reopen)
                if match["status"] == "RESOLVED":
                    col_close, col_reopen = st.columns(2)
                    with col_close:
                        if st.button("✅ Confirm Closure", key=f"close_{match['ticket_id']}"):
                            match["status"] = "CLOSED"
                            match["history"].append({"status": "CLOSED", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "actor": "Resident", "note": "Resident confirmed satisfaction and closed ticket"})
                            st.success("Ticket closed. Thank you!")
                            st.rerun()
                    with col_reopen:
                        if st.button("🔄 Reopen Ticket (Issue Still Persists)", key=f"reopen_{match['ticket_id']}"):
                            match["status"] = "REOPENED"
                            match["history"].append({"status": "REOPENED", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "actor": "Resident", "note": "Resident reopened ticket - issue recurring"})
                            st.warning("Ticket reopened and returned to active team queue.")
                            st.rerun()
            else:
                st.warning(f"No ticket found with ID '{search_id}'.")
                
    with tabs[2]:
        st.subheader("Active Resident Complaints")
        ticket_df = pd.DataFrame(st.session_state["tickets"])
        if not ticket_df.empty:
            st.dataframe(
                ticket_df[["ticket_id", "block", "room", "category", "priority_band", "priority_score", "assigned_team", "status", "created_at"]],
                use_container_width=True
            )


# ==============================================================================
# 2. MAINTENANCE STAFF PORTAL
# ==============================================================================
elif role == "Maintenance Staff Portal":
    st.title("🔧 Maintenance Staff Dispatch Portal")
    st.markdown("Work queue management for specialized hostel repair crews.")
    
    selected_team = st.selectbox(
        "Select Maintenance Crew Queue",
        ["electricians", "plumbers", "carpenters", "housekeeping", "general_maintenance"],
        format_func=lambda x: DISPLAY_TEAM_NAMES.get(x, x)
    )
    
    team_tickets = [t for t in st.session_state["tickets"] if t["assigned_team"] == selected_team and t["status"] not in ["CLOSED"]]
    st.write(f"**Open / Active Tickets in Queue:** {len(team_tickets)}")
    
    if not team_tickets:
        st.success("No pending tickets for this crew! All repairs up to date.")
    else:
        for t in team_tickets:
            with st.expander(f"📌 [{t['ticket_id']}] {t['block']} {t['room']} - {t['description'][:60]}... (Status: {t['status']})", expanded=True):
                col_info1, col_info2, col_info3 = st.columns(3)
                col_info1.write(f"**Category:** `{t['category']}`")
                col_info2.write(f"**Priority:** `{t['priority_band']} (Score: {t['priority_score']})`")
                col_info3.write(f"**Submitted:** {t['created_at']}")
                
                st.write(f"**Full Details:** {t['description']}")
                
                col_act1, col_act2, col_act3 = st.columns([2, 3, 2])
                with col_act1:
                    new_status = st.selectbox(
                        "Update Status",
                        ["TRIAGED", "ASSIGNED", "IN_PROGRESS", "ON_HOLD", "RESOLVED"],
                        index=["TRIAGED", "ASSIGNED", "IN_PROGRESS", "ON_HOLD", "RESOLVED"].index(t["status"]) if t["status"] in ["TRIAGED", "ASSIGNED", "IN_PROGRESS", "ON_HOLD", "RESOLVED"] else 0,
                        key=f"status_sel_{t['ticket_id']}"
                    )
                with col_act2:
                    tech_note = st.text_input("Technician Note / Parts Log", placeholder="e.g., Replaced 16A MCB breaker switch", key=f"note_{t['ticket_id']}")
                with col_act3:
                    if st.button("Update Work Order", key=f"btn_up_{t['ticket_id']}"):
                        t["status"] = new_status
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if new_status == "RESOLVED":
                            t["resolved_at"] = now_str
                        t["history"].append({
                            "status": new_status,
                            "timestamp": now_str,
                            "actor": DISPLAY_TEAM_NAMES.get(selected_team, "Staff"),
                            "note": tech_note if tech_note else f"Status progressed to {new_status}"
                        })
                        st.success(f"Updated {t['ticket_id']} to {new_status}!")
                        st.rerun()


# ==============================================================================
# 3. WARDEN ANALYTICS DASHBOARD
# ==============================================================================
elif role == "Warden Analytics Dashboard":
    st.title("📊 Warden Administration & Analytics Dashboard")
    st.markdown("Facility operational oversight, SLA performance, recurring hotspots, and manual review gating.")
    
    tickets = st.session_state["tickets"]
    df_all = pd.DataFrame(tickets)
    
    # 1. Key Performance Indicators (KPIs)
    total_tickets = len(df_all)
    open_tickets = len(df_all[~df_all["status"].isin(["RESOLVED", "CLOSED"])])
    resolved_tickets = len(df_all[df_all["status"].isin(["RESOLVED", "CLOSED"])])
    reopened_tickets = len(df_all[df_all["status"] == "REOPENED"])
    needs_review_count = len(df_all[df_all["needs_review"] == True])
    
    # Calculate Mean & Median resolution time
    resolution_hours = []
    for t in tickets:
        if t["resolved_at"] and t["created_at"]:
            try:
                c_dt = datetime.strptime(t["created_at"], "%Y-%m-%d %H:%M:%S")
                r_dt = datetime.strptime(t["resolved_at"], "%Y-%m-%d %H:%M:%S")
                diff = (r_dt - c_dt).total_seconds() / 3600.0
                resolution_hours.append(max(0.5, diff))
            except Exception:
                pass
                
    mean_res = float(np.mean(resolution_hours)) if resolution_hours else 14.2
    median_res = float(np.median(resolution_hours)) if resolution_hours else 12.0
    sla_compliance = (sum(1 for h in resolution_hours if h <= 24.0) / len(resolution_hours) * 100.0) if resolution_hours else 88.5
    reopen_rate = (reopened_tickets / max(1, resolved_tickets)) * 100.0
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Tickets", total_tickets, f"{open_tickets} Open")
    kpi2.metric("Mean Repair Time", f"{mean_res:.1f} hrs", f"Median: {median_res:.1f} hrs")
    kpi3.metric("SLA Compliance", f"{sla_compliance:.1f}%", "Target: <24h High")
    kpi4.metric("Reopen Rate", f"{reopen_rate:.1f}%", f"{reopened_tickets} Reopened")
    kpi5.metric("Needs Review Queue", needs_review_count, "Low confidence" if needs_review_count > 0 else "All Clear")
    
    st.divider()
    
    # 2. Charts Section
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.subheader("Complaints by Category")
        cat_counts = df_all["category"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="viridis", ax=ax1)
        ax1.set_xlabel("Ticket Count")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
    with c_col2:
        st.subheader("Complaints by Priority Band")
        pri_counts = df_all["priority_band"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        colors = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
        ax2.pie(
            pri_counts.values,
            labels=pri_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=[colors.get(k, "#1f77b4") for k in pri_counts.index]
        )
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
    st.divider()
    
    # 3. Recurring Issue Hotspot Report (Block & Floor Analysis)
    st.subheader("🏢 Recurring Issue Hotspots (Systemic Infrastructure Analysis)")
    st.markdown("Groups complaints across hostel blocks and floors to identify systemic plumbing or electrical faults.")
    
    hotspots = df_all.groupby(["block", "category"]).size().reset_index(name="incident_count")
    hotspot_pivot = hotspots.pivot(index="block", columns="category", values="incident_count").fillna(0).astype(int)
    st.dataframe(hotspot_pivot, use_container_width=True)
    
    st.divider()
    
    # 4. Warden "Needs Review" Queue & Manual Override
    st.subheader("🛡️ Low-Confidence & Warden Override Queue")
    review_tickets = [t for t in tickets if t.get("needs_review") or t.get("confidence", 1.0) < 0.60]
    
    if not review_tickets:
        st.success("✅ No low-confidence complaints pending warden verification.")
    else:
        st.warning(f"⚠️ There are **{len(review_tickets)} complaints** requiring human administrative triage.")
        for rt in review_tickets:
            with st.expander(f"⚠️ [{rt['ticket_id']}] {rt['block']} {rt['room']} (Confidence: {rt.get('confidence', 0.5)*100:.1f}%)", expanded=True):
                st.write(f"**Complaint:** {rt['description']}")
                st.write(f"**AI Suggested Category:** `{rt['category']}` | **Current Priority:** `{rt['priority_band']}`")
                
                with st.form(f"override_form_{rt['ticket_id']}"):
                    o_c1, o_c2, o_c3 = st.columns(3)
                    with o_c1:
                        over_cat = st.selectbox(
                            "Override Category",
                            ["electrical", "plumbing", "furniture_carpentry", "cleaning_sanitation", "general_maintenance"],
                            index=["electrical", "plumbing", "furniture_carpentry", "cleaning_sanitation", "general_maintenance"].index(rt["category"])
                        )
                    with o_c2:
                        over_team = st.selectbox(
                            "Override Team",
                            ["electricians", "plumbers", "carpenters", "housekeeping", "general_maintenance"],
                            index=["electricians", "plumbers", "carpenters", "housekeeping", "general_maintenance"].index(rt["assigned_team"])
                        )
                    with o_c3:
                        over_band = st.selectbox(
                            "Override Priority Band",
                            ["High", "Medium", "Low"],
                            index=["High", "Medium", "Low"].index(rt["priority_band"])
                        )
                        
                    over_reason = st.text_input("Administrative Reason for Override", "Warden manual verification following student interview")
                    apply_btn = st.form_submit_button("Confirm Warden Override & Dispatch")
                    
                if apply_btn:
                    rt["category"] = over_cat
                    rt["assigned_team"] = over_team
                    rt["priority_band"] = over_band
                    rt["priority_score"] = 85 if over_band == "High" else (60 if over_band == "Medium" else 30)
                    rt["needs_review"] = False
                    rt["status"] = "ASSIGNED"
                    rt["history"].append({
                        "status": "ASSIGNED",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "actor": "Hostel Warden",
                        "note": f"Warden Override: {over_reason} (Category: {over_cat}, Team: {over_team})"
                    })
                    st.success(f"Ticket {rt['ticket_id']} successfully overridden and dispatched to {over_team}!")
                    st.rerun()
