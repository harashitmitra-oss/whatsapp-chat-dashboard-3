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
from collections import Counter
from fpdf import FPDF
from docx import Document

# ✅ OpenAI (v1.x)
from openai import OpenAI

# ✅ All-countries phone → country detection
import phonenumbers
from phonenumbers import geocoder

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="WhatsApp Intelligence Dashboard", layout="wide")

col1, col2 = st.columns([5, 1])
with col1:
    st.title("📊 WhatsApp Intelligence & Engagement Dashboard")
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)

st.markdown(
    """
<style>
body { background-color: #ffffff; }
[data-testid="stSidebar"] { background-color: #0b3d2e; color: white; }
h1, h2, h3, h4, h5, h6 { color: #0b3d2e; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Sidebar: Settings
# -------------------------------
st.sidebar.title("⚙️ Settings")
OPENAI_API_KEY = st.sidebar.text_input("🔐 Enter OpenAI API Key", type="password").strip()

@st.cache_resource(show_spinner=False)
def get_ai_client(api_key: str):
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

ai_client = get_ai_client(OPENAI_API_KEY)

def generate_ai_summary(prompt: str):
    if not ai_client:
        return "⚠️ Please enter a valid OpenAI API key in the sidebar."
    try:
        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business intelligence analyst summarizing WhatsApp group discussions."},
                {"role": "user", "content": prompt[:12000]},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI summary unavailable: {type(e).__name__}: {e}"

# -------------------------------
# Utility Functions
# -------------------------------
def clean_phone_number(num):
    return re.sub(r"\D", "", str(num))[-10:]

def analyze_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def extract_emojis(text):
    return [char for char in str(text) if char in emoji.EMOJI_DATA]

def detect_country(sender: str):
    s = str(sender).strip()
    if not re.search(r"\+?\d", s):
        return "Unknown"
    try:
        if not s.startswith("+"):
            return "Unknown"
        num = phonenumbers.parse(s, None)
        if not phonenumbers.is_valid_number(num):
            return "Unknown"
        country = geocoder.description_for_number(num, "en")
        return country if country else "Unknown"
    except:
        return "Unknown"

def assign_engagement_level_by_percentile(series, high_q=0.80, low_q=0.20):
    if series.empty:
        return series
    hi = series.quantile(high_q)
    lo = series.quantile(low_q)

    def label(x):
        if x >= hi:
            return "High"
        elif x <= lo:
            return "Low"
        return "Medium"

    return series.apply(label)

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
            if current_message and line.strip():
                current_message[2] += " " + line.strip()

    df = pd.DataFrame(data, columns=["DateTime", "Sender", "Message"])
    df = df.dropna(subset=["DateTime"])
    df["Date"] = df["DateTime"].dt.date
    df["Time"] = df["DateTime"].dt.time
    df["Hour"] = df["DateTime"].dt.hour
    df["Week"] = df["DateTime"].dt.to_period("W").astype(str)
    df["Month"] = df["DateTime"].dt.to_period("M").astype(str)
    return df

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
# Sidebar: Group + Window
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

if not chat_file:
    st.info("📥 Please upload a WhatsApp chat file to begin analysis.")
    st.stop()

# -------------------------------
# Parse + Enrich
# -------------------------------
df = parse_whatsapp_chat(chat_file)

# Phone Number → Name Matching
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

df["Sentiment"] = df["Message"].apply(analyze_sentiment)
df["Emojis"] = df["Message"].apply(extract_emojis)

# -------------------------------
# Analysis Window Selector
# -------------------------------
st.sidebar.header("🕒 Analysis Window")
window_mode = st.sidebar.selectbox("Choose Window", ["All", "Daily", "Weekly", "Monthly"])

window_value = None
if window_mode == "Daily":
    window_value = st.sidebar.selectbox("Pick Date", sorted(df["Date"].dropna().unique()))
elif window_mode == "Weekly":
    window_value = st.sidebar.selectbox("Pick Week", sorted(df["Week"].dropna().unique()))
elif window_mode == "Monthly":
    window_value = st.sidebar.selectbox("Pick Month", sorted(df["Month"].dropna().unique()))

df_window = df.copy()
if window_mode == "Daily":
    df_window = df[df["Date"] == window_value]
elif window_mode == "Weekly":
    df_window = df[df["Week"] == window_value]
elif window_mode == "Monthly":
    df_window = df[df["Month"] == window_value]

if df_window.empty:
    st.warning("No messages found in the selected window.")
    st.stop()

# -------------------------------
# Core Metrics (Window-based)
# -------------------------------
total_messages = len(df_window)
active_members = df_window["DisplayName"].nunique()
total_members_input = st.sidebar.number_input("Total Members in Group", min_value=1, value=int(active_members))
silent_members = max(int(total_members_input) - int(active_members), 0)
activation_rate = round((active_members / total_members_input) * 100, 2)

# -------------------------------
# Participant Aggregation (Percent-based Engagement)
# -------------------------------
total_msgs_window = len(df_window)
total_days_window = df_window["Date"].nunique()
total_weeks_window = df_window["Week"].nunique()
total_months_window = df_window["Month"].nunique()

participant_stats = df_window.groupby("DisplayName").agg(
    MessageCount=("Message", "count"),
    ActiveDays=("Date", "nunique"),
    ActiveWeeks=("Week", "nunique"),
    ActiveMonths=("Month", "nunique"),
    AvgSentiment=("Sentiment", lambda x: x.mode()[0] if not x.mode().empty else "Neutral"),
).reset_index()

participant_stats["MessageSharePct"] = (participant_stats["MessageCount"] / max(total_msgs_window, 1)) * 100
participant_stats["ActiveDaysPct"] = (participant_stats["ActiveDays"] / max(total_days_window, 1)) * 100

if window_mode in ["All", "Monthly"]:
    participant_stats["ConsistencyPct"] = (participant_stats["ActiveWeeks"] / max(total_weeks_window, 1)) * 100
elif window_mode == "Weekly":
    participant_stats["ConsistencyPct"] = participant_stats["ActiveDaysPct"]
else:
    participant_stats["ConsistencyPct"] = participant_stats["ActiveDaysPct"]

participant_stats["EngagementIndex"] = (
    0.55 * participant_stats["MessageSharePct"] +
    0.30 * participant_stats["ActiveDaysPct"] +
    0.15 * participant_stats["ConsistencyPct"]
).round(2)

participant_stats["Sentiment"] = participant_stats["AvgSentiment"]
participant_stats["EngagementLevel"] = assign_engagement_level_by_percentile(participant_stats["EngagementIndex"])

def calculate_lead_score_v2(row):
    score = row["EngagementIndex"]
    if row["Sentiment"] == "Positive":
        score += 15
    elif row["Sentiment"] == "Negative":
        score -= 10
    if row["EngagementLevel"] == "High":
        score += 15
    elif row["EngagementLevel"] == "Medium":
        score += 5
    return round(score, 2)

participant_stats["LeadScore"] = participant_stats.apply(calculate_lead_score_v2, axis=1)
top_lead = participant_stats.sort_values("LeadScore", ascending=False).iloc[0]["DisplayName"] if len(participant_stats) else "N/A"

save_daily_report(group_name, participant_stats)

# -------------------------------
# KPI Panel
# -------------------------------
st.subheader("📌 Key Metrics (Selected Window)")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Activation Rate (%)", activation_rate)
col2.metric("Total Messages", total_messages)
col3.metric("Active Members", active_members)
col4.metric("Silent Members", silent_members)
col5.metric("Top Lead", top_lead)

# -------------------------------
# Engagement Pies
# -------------------------------
st.subheader("📊 Engagement Contribution (Percent-based)")

eng_msg = participant_stats.groupby("EngagementLevel")["MessageSharePct"].sum().reset_index()
fig_eng_share = px.pie(eng_msg, names="EngagementLevel", values="MessageSharePct", hole=0.45,
                       title="Message Share by Engagement Level (%)")
fig_eng_share.update_traces(textinfo="percent+label")
st.plotly_chart(fig_eng_share, use_container_width=True)

eng_index = participant_stats.groupby("EngagementLevel")["EngagementIndex"].sum().reset_index()
eng_index["IndexSharePct"] = (eng_index["EngagementIndex"] / max(eng_index["EngagementIndex"].sum(), 1)) * 100

fig_eng_index = px.pie(eng_index, names="EngagementLevel", values="IndexSharePct", hole=0.45,
                       title="Engagement Index Share by Level (%)")
fig_eng_index.update_traces(textinfo="percent+label")
st.plotly_chart(fig_eng_index, use_container_width=True)

# -------------------------------
# Users by Engagement Level + AI
# -------------------------------
st.subheader("👥 Users by Engagement Level")

for level in ["High", "Medium", "Low"]:
    st.markdown(f"### {level} Engagement Users")
    subset = participant_stats[participant_stats["EngagementLevel"] == level].sort_values("EngagementIndex", ascending=False)
    st.dataframe(subset[["DisplayName", "MessageCount", "MessageSharePct", "ActiveDaysPct", "EngagementIndex",
                         "Sentiment", "LeadScore"]], use_container_width=True)

    subset_df = df_window[df_window["DisplayName"].isin(subset["DisplayName"])]
    if len(subset_df) > 0:
        sample_size = min(40, len(subset_df))
        sample_msgs = subset_df.sort_values("DateTime").tail(sample_size)["Message"].tolist()
        msgs_text = "\n- " + "\n- ".join([str(m) for m in sample_msgs if str(m).strip()][:40])
    else:
        msgs_text = ""

    ai_prompt = f"""
Analyze the following WhatsApp messages from {level} engagement users.
Summarize key themes, concerns, interests, and any business signals.
Write business-friendly bullet points.

Messages:{msgs_text}
"""
    st.info(f"🤖 AI Insight on {level} Engagement Users:\n\n{generate_ai_summary(ai_prompt)}")

# -------------------------------
# Sentiment Distribution (Window)
# -------------------------------
st.subheader("💬 Sentiment Distribution (Selected Window)")
sent_counts = df_window["Sentiment"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
sent_counts["Pct"] = (sent_counts["Count"] / sent_counts["Count"].sum()) * 100

fig_sent = px.bar(sent_counts, x="Sentiment", y="Pct", title="Sentiment Distribution (%)", text=sent_counts["Pct"].round(1))
st.plotly_chart(fig_sent, use_container_width=True)

# -------------------------------
# Emoji Analysis (Window)
# -------------------------------
st.subheader("😄 Emoji Usage Analysis (Selected Window)")
all_emojis = [e for sublist in df_window["Emojis"] for e in sublist]
emoji_counts = Counter(all_emojis)

emoji_df = pd.DataFrame(emoji_counts.most_common(15), columns=["Emoji", "Count"])
if not emoji_df.empty:
    emoji_df["Pct"] = (emoji_df["Count"] / emoji_df["Count"].sum()) * 100
    fig_emoji = px.bar(emoji_df, x="Emoji", y="Pct", title="Top Emojis Used (%)", text=emoji_df["Pct"].round(1))
    st.plotly_chart(fig_emoji, use_container_width=True)
else:
    st.info("No emojis found in the selected window.")

st.subheader("📈 Emoji Sentiment Trend Over Time (Window)")
df_exploded = df_window.explode("Emojis")
emoji_sentiment_df = df_exploded.groupby(["Date", "Sentiment"]).size().reset_index(name="Count")
if not emoji_sentiment_df.empty:
    fig_emoji_trend = px.line(emoji_sentiment_df, x="Date", y="Count", color="Sentiment",
                              title="Emoji-Related Sentiment Trend Over Time")
    st.plotly_chart(fig_emoji_trend, use_container_width=True)
else:
    st.info("Not enough emoji data for trend plotting.")

# -------------------------------
# Topic / Discussion Analysis (Window)
# -------------------------------
st.subheader("🧠 Topic & Discussion Analysis (Selected Window)")
all_text = " ".join(df_window["Message"].astype(str))
if all_text.strip():
    wc = WordCloud(width=900, height=320, background_color="white").generate(all_text)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 4))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

topic_msgs = df_window["Message"].astype(str).sample(min(250, len(df_window)), random_state=42).tolist()
topic_text = "\n- " + "\n- ".join(topic_msgs[:250])

topic_prompt = f"""
Analyze these WhatsApp group messages and extract:
1) Main topics discussed
2) Key recurring themes
3) Opportunities / risks / concerns
4) Suggested actions for business stakeholders

Messages:{topic_text}
"""
st.info(f"🤖 AI Topic & Theme Analysis:\n\n{generate_ai_summary(topic_prompt)}")

# -------------------------------
# Participant Search & Profile
# -------------------------------
st.subheader("🔍 Participant-Level Analytics")

# ✅ IMPORTANT UPDATE: choose from FULL chat users, and timeline uses FULL chat
selected_user = st.selectbox("Select a participant", sorted(df["DisplayName"].unique()))

# Window-based stats for KPIs (if user not in window, show zeros)
user_stats_window = participant_stats[participant_stats["DisplayName"] == selected_user]
if not user_stats_window.empty:
    user_stats = user_stats_window.iloc[0]
else:
    user_stats = {
        "MessageCount": 0,
        "MessageSharePct": 0.0,
        "EngagementIndex": 0.0,
        "EngagementLevel": "N/A",
        "LeadScore": 0.0
    }

# ✅ FULL chat data for timeline, irrespective of window
user_df = df[df["DisplayName"] == selected_user]

colu1, colu2, colu3, colu4, colu5 = st.columns(5)
colu1.metric("Message Count (Window)", int(user_stats["MessageCount"]))
colu2.metric("Message Share % (Window)", round(float(user_stats["MessageSharePct"]), 2))
colu3.metric("Engagement Index (Window)", round(float(user_stats["EngagementIndex"]), 2))
colu4.metric("Engagement Level (Window)", user_stats["EngagementLevel"])
colu5.metric("Lead Score (Window)", round(float(user_stats["LeadScore"]), 2))

st.subheader("📅 User Engagement Timeline (Full Chat)")
timeline_option = st.radio("View by:", ["Daily", "Weekly", "Monthly"], horizontal=True)

if timeline_option == "Daily":
    timeline_data = user_df.groupby("Date").size().reset_index(name="Messages")
    fig_user_timeline = px.line(timeline_data, x="Date", y="Messages", title=f"{selected_user} - Daily Engagement (Full Chat)")
elif timeline_option == "Weekly":
    timeline_data = user_df.groupby("Week").size().reset_index(name="Messages")
    fig_user_timeline = px.line(timeline_data, x="Week", y="Messages", title=f"{selected_user} - Weekly Engagement (Full Chat)")
else:
    timeline_data = user_df.groupby("Month").size().reset_index(name="Messages")
    fig_user_timeline = px.line(timeline_data, x="Month", y="Messages", title=f"{selected_user} - Monthly Engagement (Full Chat)")

st.plotly_chart(fig_user_timeline, use_container_width=True)

# -------------------------------
# Overall Engagement Trends (Full Chat)
# -------------------------------
st.subheader("📈 Overall Engagement Trends (Full Chat)")
daily_trend = df.groupby("Date").size().reset_index(name="Messages")
weekly_trend = df.groupby("Week").size().reset_index(name="Messages")
monthly_trend = df.groupby("Month").size().reset_index(name="Messages")

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    st.plotly_chart(px.line(daily_trend, x="Date", y="Messages", title="Daily Engagement (Full Chat)"), use_container_width=True)
with col_t2:
    st.plotly_chart(px.line(weekly_trend, x="Week", y="Messages", title="Weekly Engagement (Full Chat)"), use_container_width=True)
with col_t3:
    st.plotly_chart(px.line(monthly_trend, x="Month", y="Messages", title="Monthly Engagement (Full Chat)"), use_container_width=True)

# -------------------------------
# Heatmap (Window)
# -------------------------------
st.subheader("🔥 Activity Heatmap (Day vs Hour) (Selected Window)")
df_window["DayName"] = df_window["DateTime"].dt.day_name()
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_data = df_window.pivot_table(index="DayName", columns="Hour", values="Message", aggfunc="count").fillna(0)
heatmap_data = heatmap_data.reindex(day_order).dropna(how="all")

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(heatmap_data.values, aspect="auto", cmap="Blues")
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(list(heatmap_data.columns))
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(list(heatmap_data.index))
plt.colorbar(im)
st.pyplot(fig)

# -------------------------------
# Country-Level Engagement (Window)
# -------------------------------
st.subheader("🌍 Country-Level Engagement (Selected Window)")
country_counts = df_window["Country"].value_counts().reset_index()
country_counts.columns = ["Country", "Messages"]

hide_unknown = st.checkbox("Hide 'Unknown' country", value=False)
if hide_unknown:
    country_counts = country_counts[country_counts["Country"] != "Unknown"]

if not country_counts.empty:
    fig_country = px.bar(country_counts, x="Country", y="Messages", title="Messages by Country (Window)")
    st.plotly_chart(fig_country, use_container_width=True)
else:
    st.info("No country data available (numbers may not be in international + format).")

# -------------------------------
# Group Comparison Dashboard (Historical)
# -------------------------------
st.subheader("🆚 Group Comparison Dashboard (Historical)")
historical_df = load_historical_reports()
if not historical_df.empty:
    hist_summary = historical_df.groupby("SourceFile").agg(
        AvgMessages=("MessageCount", "mean"),
        AvgLeadScore=("LeadScore", "mean"),
        Users=("DisplayName", "nunique")
    ).reset_index()

    fig_group_compare = px.bar(hist_summary, x="SourceFile", y="AvgMessages", color="AvgLeadScore",
                               title="Group-wise Avg Messages vs Avg Lead Score")
    st.plotly_chart(fig_group_compare, use_container_width=True)
else:
    st.info("No historical group data available yet.")

# -------------------------------
# AI Summary (Window)
# -------------------------------
st.subheader("🤖 AI Summary (Selected Window)")
sample_msgs = df_window.sort_values("DateTime").tail(min(350, len(df_window)))["Message"].astype(str).tolist()
sample_text = "\n- " + "\n- ".join(sample_msgs)

ai_prompt_daily = f"""
Summarize the selected WhatsApp conversation window for business stakeholders.
Include:
- Overall tone
- Key topics
- Engagement quality
- Opportunities / risks
- Suggested next actions

Messages:{sample_text}
"""
ai_daily_summary = generate_ai_summary(ai_prompt_daily)
st.success(ai_daily_summary)

st.subheader("🧠 AI Insights & High Intent Signals (Selected Window)")
ai_prompt_insights = f"""
Analyze these WhatsApp messages and identify:
1) Emerging trends
2) Issues/complaints
3) Opportunities for conversion
4) Users showing high buying intent (and why)

Messages:{sample_text}
"""
st.info(generate_ai_summary(ai_prompt_insights))

# -------------------------------
# Automated Report Generation
# -------------------------------
st.subheader("📄 Automated Report Generation")
metrics_dict = {
    "Window": f"{window_mode} {window_value if window_value else ''}".strip(),
    "Activation Rate (%)": activation_rate,
    "Total Messages": total_messages,
    "Active Members": active_members,
    "Silent Members": silent_members,
    "Top Lead": top_lead,
}

if st.button("⬇️ Download PDF Report"):
    pdf_file = generate_pdf_report(ai_daily_summary, metrics_dict, filename="whatsapp_report.pdf")
    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF", f, file_name="whatsapp_report.pdf")

if st.button("⬇️ Download Word Report"):
    word_file = generate_word_report(ai_daily_summary, metrics_dict, filename="whatsapp_report.docx")
    with open(word_file, "rb") as f:
        st.download_button("Download Word", f, file_name="whatsapp_report.docx")
