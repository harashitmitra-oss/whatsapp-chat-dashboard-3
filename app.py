import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
from datetime import datetime
import emoji
from collections import Counter, defaultdict
from fpdf import FPDF
from docx import Document
from openai import OpenAI

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="📊 WhatsApp Intelligence Dashboard", layout="wide")

# -------------------------------
# Sidebar: AI Key & Settings
# -------------------------------
st.sidebar.title("⚙️ Settings")
OPENAI_API_KEY = st.sidebar.text_input("🔐 Enter OpenAI API Key", type="password")
ai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------
# Utility Functions
# -------------------------------
def generate_ai_summary(prompt):
    if not ai_client:
        return "⚠️ Please enter your OpenAI API key in the sidebar."
    try:
        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business intelligence analyst summarizing WhatsApp group discussions."},
                {"role": "user", "content": prompt[:7000]}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI summary unavailable: {e}"

def clean_phone_number(num):
    return re.sub(r"\D", "", str(num))[-10:]

# -------------------------------
# WhatsApp Chat Parser (Robust)
# -------------------------------
def parse_whatsapp_chat(file):
    content = file.read()
    try:
        content = content.decode("utf-8")
    except:
        content = content.decode("latin-1")

    lines = content.split("\n")
    data = []
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})(?: (AM|PM))? - (.*?): (.*)"

    current_message = None

    for line in lines:
        match = re.match(pattern, line)
        if match:
            date, time, meridiem, sender, message = match.groups()
            timestamp_str = f"{date} {time} {meridiem}" if meridiem else f"{date} {time}"
            timestamp = pd.to_datetime(timestamp_str, dayfirst=True, errors="coerce")
            current_message = [timestamp, sender.strip(), message.strip()]
            data.append(current_message)
        else:
            if current_message:
                current_message[2] += " " + line.strip()

    df = pd.DataFrame(data, columns=["DateTime", "Sender", "Message"])
    df["Date"] = df["DateTime"].dt.date
    df["Time"] = df["DateTime"].dt.time
    df["Hour"] = df["DateTime"].dt.hour
    df["Week"] = df["DateTime"].dt.to_period("W").astype(str)
    df["Month"] = df["DateTime"].dt.to_period("M").astype(str)
    return df

# -------------------------------
# Sentiment Analysis
# -------------------------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# -------------------------------
# Engagement Level Categorization
# -------------------------------
def categorize_engagement(msg_count):
    if msg_count >= 50:
        return "High"
    elif msg_count >= 15:
        return "Medium"
    else:
        return "Low"

# -------------------------------
# Lead Scoring System
# -------------------------------
def calculate_lead_score(row):
    score = row["MessageCount"] * 1.2
    if row["Sentiment"] == "Positive":
        score += 20
    elif row["Sentiment"] == "Negative":
        score -= 10
    if row["EngagementLevel"] == "High":
        score += 25
    elif row["EngagementLevel"] == "Medium":
        score += 10
    return round(score, 2)

# -------------------------------
# Emoji Analysis
# -------------------------------
def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

# -------------------------------
# Country Detection (Simple Heuristic)
# -------------------------------
def detect_country(sender):
    if sender.startswith("+91"):
        return "India"
    elif sender.startswith("+1"):
        return "USA"
    elif sender.startswith("+44"):
        return "UK"
    elif sender.startswith("+61"):
        return "Australia"
    else:
        return "Unknown"

# -------------------------------
# Report Storage
# -------------------------------
DATA_DIR = "stored_reports"
os.makedirs(DATA_DIR, exist_ok=True)

def save_daily_report(group_name, df_summary):
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(DATA_DIR, f"{group_name}_{date_str}.csv")
    df_summary.to_csv(file_path, index=False)

def load_historical_reports():
    files = os.listdir(DATA_DIR)
    data = []
    for f in files:
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            df["SourceFile"] = f
            data.append(df)
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

# -------------------------------
# PDF & Word Report Generation
# -------------------------------
def generate_pdf_report(summary_text, metrics_dict, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, "WhatsApp Group Intelligence Report\n\n")
    for k, v in metrics_dict.items():
        pdf.multi_cell(0, 6, f"{k}: {v}")
    pdf.multi_cell(0, 6, "\nAI Summary:\n")
    pdf.multi_cell(0, 6, summary_text)
    pdf.output(filename)
    return filename

def generate_word_report(summary_text, metrics_dict, filename="report.docx"):
    doc = Document()
    doc.add_heading("WhatsApp Group Intelligence Report", level=1)
    for k, v in metrics_dict.items():
        doc.add_paragraph(f"{k}: {v}")
    doc.add_heading("AI Summary", level=2)
    doc.add_paragraph(summary_text)
    doc.save(filename)
    return filename

# -------------------------------
# UI STARTS HERE
# -------------------------------
st.title("📊 WhatsApp Intelligence & Engagement Dashboard")

# -------------------------------
# Group Management
# -------------------------------
st.sidebar.header("👥 Group Management")
group_name = st.sidebar.text_input("Enter Group Name", value="Default_Group")

# -------------------------------
# Upload Files
# -------------------------------
st.header("📂 Upload WhatsApp Chat File")
chat_file = st.file_uploader("Upload WhatsApp .txt chat file", type=["txt"])

st.header("📑 Upload Conversion Tracker (Optional)")
excel_file = st.file_uploader("Upload Excel file with names (multi-sheet supported)", type=["xlsx"])

if chat_file:
    df = parse_whatsapp_chat(chat_file)

    # -------------------------------
    # Phone Number → Name Matching
    # -------------------------------
    number_name_map = {}
    if excel_file:
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        for sheet, data in excel_data.items():
            for col in data.columns:
                if "phone" in col.lower() or "mobile" in col.lower():
                    name_col = [c for c in data.columns if "name" in c.lower()]
                    if name_col:
                        for _, row in data.iterrows():
                            num = clean_phone_number(row[col])
                            name = str(row[name_col[0]])
                            if num:
                                number_name_map[num] = name

    def map_sender(sender):
        num = clean_phone_number(sender)
        return number_name_map.get(num, sender)

    df["DisplayName"] = df["Sender"].apply(map_sender)
    df["Country"] = df["Sender"].apply(detect_country)

    # -------------------------------
    # Sentiment
    # -------------------------------
    df["Sentiment"] = df["Message"].apply(analyze_sentiment)

    # -------------------------------
    # Emoji Extraction
    # -------------------------------
    df["Emojis"] = df["Message"].apply(extract_emojis)
    all_emojis = [e for sublist in df["Emojis"] for e in sublist]
    emoji_counts = Counter(all_emojis)

    # -------------------------------
    # Core Metrics
    # -------------------------------
    total_messages = len(df)
    active_members = df["DisplayName"].nunique()

    total_members_input = st.sidebar.number_input("Total Members in Group", min_value=1, value=active_members)
    silent_members = total_members_input - active_members
    activation_rate = round((active_members / total_members_input) * 100, 2)

    # -------------------------------
    # Participant-Level Aggregation
    # -------------------------------
    participant_stats = df.groupby("DisplayName").agg(
        MessageCount=("Message", "count"),
        AvgSentiment=("Sentiment", lambda x: x.mode()[0] if not x.mode().empty else "Neutral")
    ).reset_index()

    participant_stats["EngagementLevel"] = participant_stats["MessageCount"].apply(categorize_engagement)
    participant_stats["Sentiment"] = participant_stats["AvgSentiment"]
    participant_stats["LeadScore"] = participant_stats.apply(calculate_lead_score, axis=1)

    top_lead = participant_stats.sort_values("LeadScore", ascending=False).iloc[0]["DisplayName"]

    # -------------------------------
    # Save Daily Summary
    # -------------------------------
    save_daily_report(group_name, participant_stats)

    # -------------------------------
    # KPI Panel
    # -------------------------------
    st.subheader("📌 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Activation Rate (%)", activation_rate)
    col2.metric("Total Messages", total_messages)
    col3.metric("Active Members", active_members)
    col4.metric("Silent Members", silent_members)
    col5.metric("Top Lead", top_lead)

    # -------------------------------
    # Engagement Distribution
    # -------------------------------
    st.subheader("📊 Engagement Level Distribution")
    eng_counts = participant_stats["EngagementLevel"].value_counts().reset_index()
    eng_counts.columns = ["EngagementLevel", "Count"]
    fig_eng = px.pie(eng_counts, names="EngagementLevel", values="Count", hole=0.4, title="Engagement Distribution")
    st.plotly_chart(fig_eng, use_container_width=True)

    # -------------------------------
    # List Users by Engagement Level + AI Explanation
    # -------------------------------
    st.subheader("👥 Users by Engagement Level")
    for level in ["High", "Medium", "Low"]:
        st.markdown(f"### {level} Engagement Users")
        subset = participant_stats[participant_stats["EngagementLevel"] == level].sort_values("MessageCount", ascending=False)
        st.dataframe(subset[["DisplayName", "MessageCount", "Sentiment", "LeadScore"]])

        # AI explanation
        subset_df = df[df["DisplayName"].isin(subset["DisplayName"])]
sample_msgs = subset_df.sample(min(30, len(subset_df)), random_state=42)["Message"].tolist() if len(subset_df) > 0 else []

        ai_prompt = f"""
These are messages from {level} engagement users in a WhatsApp group. 
Summarize what they mostly talk about and their intent in business terms:

Messages:
{sample_msgs}
"""
        ai_text = generate_ai_summary(ai_prompt)
        st.info(f"🤖 AI Insight on {level} Engagement Users:\n\n{ai_text}")

    # -------------------------------
    # Sentiment Distribution
    # -------------------------------
    st.subheader("💬 Sentiment Distribution")
    sent_counts = df["Sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    fig_sent = px.bar(sent_counts, x="Sentiment", y="Count", title="Sentiment Distribution", text="Count")
    st.plotly_chart(fig_sent, use_container_width=True)

    # -------------------------------
    # Emoji Analysis + Emoji Sentiment Trend
    # -------------------------------
    st.subheader("😄 Emoji Usage Analysis")
    emoji_df = pd.DataFrame(emoji_counts.most_common(15), columns=["Emoji", "Count"])
    fig_emoji = px.bar(emoji_df, x="Emoji", y="Count", title="Top Emojis Used")
    st.plotly_chart(fig_emoji, use_container_width=True)

    # Emoji sentiment trend
    st.subheader("📈 Emoji Sentiment Trend Over Time")
    df_exploded = df.explode("Emojis")
    emoji_sentiment_df = df_exploded.groupby(["Date", "Sentiment"]).size().reset_index(name="Count")
    fig_emoji_trend = px.line(emoji_sentiment_df, x="Date", y="Count", color="Sentiment",
                              title="Emoji-Based Sentiment Trend Over Time")
    st.plotly_chart(fig_emoji_trend, use_container_width=True)

    # -------------------------------
    # Topic / Discussion Analysis
    # -------------------------------
    st.subheader("🧠 Topic & Discussion Analysis")
    all_text = " ".join(df["Message"])
    wc = WordCloud(width=800, height=300, background_color="white").generate(all_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    topic_prompt = f"""
Analyze these WhatsApp group messages and extract:
1. Main topics discussed
2. Key recurring themes
3. Any business opportunities or concerns

Messages:
{df["Message"].sample(min(200, len(df)), random_state=42).tolist()}
"""
    ai_topics = generate_ai_summary(topic_prompt)
    st.info(f"🤖 AI Topic & Theme Analysis:\n\n{ai_topics}")

    # -------------------------------
    # Participant Search & Profile
    # -------------------------------
    st.subheader("🔍 Participant-Level Analytics")
    selected_user = st.selectbox("Select a participant", participant_stats["DisplayName"].sort_values())

    user_df = df[df["DisplayName"] == selected_user]
    user_stats = participant_stats[participant_stats["DisplayName"] == selected_user].iloc[0]

    colu1, colu2, colu3, colu4 = st.columns(4)
    colu1.metric("Message Count", user_stats["MessageCount"])
    colu2.metric("Engagement Level", user_stats["EngagementLevel"])
    colu3.metric("Sentiment", user_stats["Sentiment"])
    colu4.metric("Lead Score", user_stats["LeadScore"])

    # User engagement timeline
    st.subheader("📅 User Engagement Timeline")
    timeline_option = st.radio("View by:", ["Daily", "Weekly", "Monthly"], horizontal=True)

    if timeline_option == "Daily":
        timeline_data = user_df.groupby("Date").size().reset_index(name="Messages")
        fig_user_timeline = px.line(timeline_data, x="Date", y="Messages", title=f"{selected_user} - Daily Engagement")
    elif timeline_option == "Weekly":
        timeline_data = user_df.groupby("Week").size().reset_index(name="Messages")
        fig_user_timeline = px.line(timeline_data, x="Week", y="Messages", title=f"{selected_user} - Weekly Engagement")
    else:
        timeline_data = user_df.groupby("Month").size().reset_index(name="Messages")
        fig_user_timeline = px.line(timeline_data, x="Month", y="Messages", title=f"{selected_user} - Monthly Engagement")

    st.plotly_chart(fig_user_timeline, use_container_width=True)

    # -------------------------------
    # Overall Engagement Timelines
    # -------------------------------
    st.subheader("📈 Overall Engagement Trends")

    daily_trend = df.groupby("Date").size().reset_index(name="Messages")
    weekly_trend = df.groupby("Week").size().reset_index(name="Messages")
    monthly_trend = df.groupby("Month").size().reset_index(name="Messages")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.plotly_chart(px.line(daily_trend, x="Date", y="Messages", title="Daily Engagement"), use_container_width=True)
    with col_t2:
        st.plotly_chart(px.line(weekly_trend, x="Week", y="Messages", title="Weekly Engagement"), use_container_width=True)
    with col_t3:
        st.plotly_chart(px.line(monthly_trend, x="Month", y="Messages", title="Monthly Engagement"), use_container_width=True)

    # -------------------------------
    # Heatmap (Day vs Hour) without seaborn
    # -------------------------------
    st.subheader("🔥 Activity Heatmap (Day vs Hour)")
    df["DayName"] = df["DateTime"].dt.day_name()
    heatmap_data = df.pivot_table(index="DayName", columns="Hour", values="Message", aggfunc="count").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    plt.colorbar(im)
    st.pyplot(fig)

    # -------------------------------
    # Country-Level Engagement
    # -------------------------------
    st.subheader("🌍 Country-Level Engagement")
    country_counts = df["Country"].value_counts().reset_index()
    country_counts.columns = ["Country", "Messages"]
    fig_country = px.bar(country_counts, x="Country", y="Messages", title="Messages by Country")
    st.plotly_chart(fig_country, use_container_width=True)

    # -------------------------------
    # Group Comparison Dashboard (Historical)
    # -------------------------------
    st.subheader("🆚 Group Comparison Dashboard")
    historical_df = load_historical_reports()
    if not historical_df.empty:
        hist_summary = historical_df.groupby("SourceFile").agg(
            AvgMessages=("MessageCount", "mean"),
            AvgLeadScore=("LeadScore", "mean"),
            Users=("DisplayName", "nunique")
        ).reset_index()

        fig_group_compare = px.bar(hist_summary, x="SourceFile", y="AvgMessages",
                                    color="AvgLeadScore",
                                    title="Group-wise Avg Messages vs Lead Score")
        st.plotly_chart(fig_group_compare, use_container_width=True)
    else:
        st.info("No historical group data available yet.")

    # -------------------------------
    # AI-Powered Daily Summary
    # -------------------------------
    st.subheader("🤖 AI-Powered Daily Summary")
    ai_prompt_daily = f"""
Summarize today's WhatsApp group conversation for business stakeholders.
Include:
- Overall tone
- Key discussion topics
- Engagement quality
- Opportunities or risks

Messages:
{df["Message"].sample(min(300, len(df)), random_state=42).tolist()}
"""
    ai_daily_summary = generate_ai_summary(ai_prompt_daily)
    st.success(ai_daily_summary)

    # -------------------------------
    # AI-Based Insights & Lead Intent Classification
    # -------------------------------
    st.subheader("🧠 AI-Based Insights & Lead Intent Detection")
    ai_prompt_insights = f"""
Analyze these WhatsApp group messages and identify:
1. Emerging trends
2. Issues or complaints
3. Opportunities for conversion
4. Users showing high buying intent

Messages:
{df["Message"].sample(min(400, len(df)), random_state=42).tolist()}
"""
    ai_insights = generate_ai_summary(ai_prompt_insights)
    st.info(ai_insights)

    # -------------------------------
    # Automated Report Generation
    # -------------------------------
    st.subheader("📄 Automated Report Generation")

    metrics_dict = {
        "Activation Rate (%)": activation_rate,
        "Total Messages": total_messages,
        "Active Members": active_members,
        "Silent Members": silent_members,
        "Top Lead": top_lead
    }

    if st.button("⬇️ Download PDF Report"):
        pdf_file = generate_pdf_report(ai_daily_summary, metrics_dict, filename="whatsapp_report.pdf")
        with open(pdf_file, "rb") as f:
            st.download_button("Download PDF", f, file_name="whatsapp_report.pdf")

    if st.button("⬇️ Download Word Report"):
        word_file = generate_word_report(ai_daily_summary, metrics_dict, filename="whatsapp_report.docx")
        with open(word_file, "rb") as f:
            st.download_button("Download Word", f, file_name="whatsapp_report.docx")

else:
    st.info("📥 Please upload a WhatsApp chat file to begin analysis.")

