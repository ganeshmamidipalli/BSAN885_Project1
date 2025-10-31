# streamlit_app.py
# -------------------------------------------------------------------
# Chat-driven dashboard for Amazon Sports insights (subset)
# Uses: outputs/Sports_review_insights_subset_445.json
# -------------------------------------------------------------------
import os, json, re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Sports Reviews – Chat Dashboard", layout="wide")
INSIGHTS_PATH = Path(os.environ.get("INSIGHTS_PATH", "outputs/Sports_review_insights_subset_445.json"))

# -----------------------------
# Helpers
# -----------------------------
def load_insights(path: Path) -> list[dict]:
    if not path.exists():
        st.error(f"Insights file not found: {path}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def explode_aspects(records: list[dict]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(records):
        if not isinstance(r, dict) or r.get("_error"):
            continue
        stars = r.get("stars", None)
        try:
            stars = int(stars) if stars is not None else None
        except Exception:
            stars = None
        osent = (r.get("overall_sentiment") or "").strip().lower() or None
        for a in r.get("aspects", []) or []:
            rows.append({
                "row_id": i,
                "aspect": (a.get("name") or "").strip().lower(),
                "theme": ((a.get("theme") or "").strip().lower() or None),
                "sentiment": (a.get("sentiment") or "").strip().lower(),
                "intensity": float(a.get("intensity", 1.0) or 1.0),
                "evidence": (a.get("evidence") or "").strip(),
                "stars": stars,
                "overall_sentiment": osent
            })
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No aspect rows found in insights JSON.")
    # light cleanup
    if "aspect" in df:
        df["aspect"] = df["aspect"].fillna("").str.slice(0, 60)
    return df

# ---------- simple rules-based intent (fallback if no OpenAI key) ----------
def parse_intent_rules(q: str) -> dict:
    q = (q or "").lower()
    out = {"sentiments": None, "stars": None, "must_contain": [], "themes": []}

    # stars
    if "low" in q or "1-2" in q or "1 to 2" in q:
        out["stars"] = [1, 2]
    if "high" in q or "4-5" in q or "4 to 5" in q:
        out["stars"] = [4, 5]
    for s in ["1 star", "2 star", "3 star", "4 star", "5 star"]:
        if s in q:
            out["stars"] = [int(s[0])]

    # explicit sentiments
    if "negative" in q or "complaints" in q or "pain" in q:
        out["sentiments"] = ["negative"]
    if "positive" in q or "praise" in q or "love" in q:
        out["sentiments"] = ["positive"]

    # common themes/aspects vocabulary to auto-select
    vocab = [
        "fit","size","sizing","durability","quality","price","value","customer service",
        "stitching","strap","mount","battery","sensor","heart rate","comfort","materials",
        "shipping","warranty"
    ]
    # match multi-words first
    multi = [v for v in vocab if " " in v and v in q]
    single = [v for v in vocab if " " not in v and re.search(rf"\b{re.escape(v)}\b", q)]
    out["must_contain"] = list(dict.fromkeys(multi + single))

    # themes (same set for now; you can map to canonical themes if you use them)
    out["themes"] = out["must_contain"]
    return out

# ---------- optional: OpenAI intent (if OPENAI_API_KEY is set) ----------
def parse_intent_llm(q: str) -> dict | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Return ONLY JSON with keys: sentiments (list of 'positive'/'negative'), "
            "stars (list of ints), must_contain (list of strings), themes (list of strings). "
            "Infer useful defaults. Example: {\"sentiments\":[\"negative\"],\"stars\":[1,2],"
            "\"must_contain\":[\"fit\"],\"themes\":[\"fit\"]}\n\n"
            f"Query: {q}"
        )
        r = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":"Extract structured filters; keep JSON minimal."},
                {"role":"user","content": prompt}
            ]
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return None

def parse_intent(q: str) -> dict:
    return parse_intent_llm(q) or parse_intent_rules(q)

# ---------- aggregations ----------
def aggregate(df: pd.DataFrame) -> dict:
    freq = Counter(df["aspect"].dropna().tolist())
    themes = Counter(df["theme"].dropna().tolist())
    # weighted sums by intensity
    neg = df[df["sentiment"]=="negative"].groupby("aspect")["intensity"].sum().to_dict()
    pos = df[df["sentiment"]=="positive"].groupby("aspect")["intensity"].sum().to_dict()
    osent = Counter(df["overall_sentiment"].dropna().tolist())
    return {"freq": freq, "weighted_neg": neg, "weighted_pos": pos, "themes": themes, "overall_sent": osent}

def top_k_dict(d: dict, k: int = 20) -> list[tuple[str, float]]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

# ---------- charts ----------
def barh_from_pairs(pairs, title, xlab):
    if not pairs:
        st.info("No data to plot.")
        return
    labels = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    fig = px.bar(x=values, y=labels, orientation="h", title=title, labels={"x": xlab, "y": ""})
    st.plotly_chart(fig, use_container_width=True)
    return labels[::-1]  # return in original order for selection lists

# -----------------------------
# Load + transform
# -----------------------------
recs = load_insights(INSIGHTS_PATH)
df_aspects = explode_aspects(recs)

# -----------------------------
# UI — Header & Chat
# -----------------------------
st.title("Amazon Sports Reviews — Chat-Driven Dashboard")
st.caption(f"Data: {INSIGHTS_PATH}")

with st.sidebar:
    st.subheader("Ask anything")
    q = st.text_input("e.g., show negative durability issues, 4–5 stars")
    parsed = parse_intent(q) if q else {}
    st.write("Parsed filters:", parsed if parsed else "—")

    st.divider()
    st.subheader("Filters (manual)")
    sentiments = st.multiselect("Sentiment", ["negative","positive","neutral","mixed"], default=parsed.get("sentiments") or [])
    stars = st.multiselect("Stars", [1,2,3,4,5], default=parsed.get("stars") or [])
    aspects_text = st.text_input("Aspect contains… (comma-separated)", value=", ".join(parsed.get("must_contain", [])) if parsed else "")
    themes_text = st.text_input("Theme contains… (comma-separated)", value=", ".join(parsed.get("themes", [])) if parsed else "")
    min_count = st.slider("Min count to show", min_value=1, max_value=20, value=3, step=1)

# -----------------------------
# Apply filters
# -----------------------------
sub = df_aspects.copy()

if sentiments:
    sub = sub[sub["sentiment"].isin(sentiments)]

if stars:
    sub = sub[sub["stars"].isin(stars)]

def _parse_list(s): 
    return [w.strip().lower() for w in (s.split(",") if s else []) if w.strip()]

aspect_terms = _parse_list(aspects_text)
theme_terms = _parse_list(themes_text)

if aspect_terms:
    pat = "|".join([re.escape(t) for t in aspect_terms])
    sub = sub[sub["aspect"].fillna("").str.contains(pat, case=False, regex=True)]
if theme_terms:
    pat = "|".join([re.escape(t) for t in theme_terms])
    sub = sub[sub["theme"].fillna("").str.contains(pat, case=False, regex=True)]

# -----------------------------
# KPI cards
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Aspect rows", f"{len(sub):,}")
c2.metric("Unique aspects", sub["aspect"].nunique())
neg_share = (sub["sentiment"].eq("negative").mean()*100) if len(sub) else 0
c3.metric("Negative share", f"{neg_share:.1f}%")
c4.metric("With stars", sub["stars"].notna().sum())

# -----------------------------
# Aggregations & Charts
# -----------------------------
agg = aggregate(sub)

st.subheader("Top Aspects — Frequency")
freq_pairs = [(k,v) for k,v in agg["freq"].items() if v >= min_count]
labels_freq = barh_from_pairs(top_k_dict(dict(freq_pairs), 20), "Top Aspects by Frequency", "mentions")

st.subheader("Top Negative Aspects — Weighted by Intensity")
neg_pairs = [(k,v) for k,v in agg["weighted_neg"].items() if v >= min_count]
labels_neg = barh_from_pairs(top_k_dict(dict(neg_pairs), 20), "Top Negative Aspects (Weighted)", "weighted score")

st.subheader("Top Positive Aspects — Weighted by Intensity")
pos_pairs = [(k,v) for k,v in agg["weighted_pos"].items() if v >= min_count]
labels_pos = barh_from_pairs(top_k_dict(dict(pos_pairs), 20), "Top Positive Aspects (Weighted)", "weighted score")

st.subheader("Top Themes")
theme_pairs = [(k,v) for k,v in agg["themes"].items() if v >= min_count]
labels_theme = barh_from_pairs(top_k_dict(dict(theme_pairs), 20), "Top Themes", "mentions")

# Overall sentiment mix (from record-level overall_sentiment)
if agg["overall_sent"]:
    st.subheader("Overall Sentiment Mix (per review)")
    os_df = pd.DataFrame({"label": list(agg["overall_sent"].keys()),
                          "count": list(agg["overall_sent"].values())})
    fig = px.bar(os_df, x="label", y="count", title="Overall Sentiment (review-level)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Evidence Explorer
# -----------------------------
st.subheader("Evidence Explorer")
# suggest aspect options from what is currently visible
options = sorted({*(labels_freq or []), *(labels_neg or []), *(labels_pos or [])})
aspect_pick = st.selectbox("Pick an aspect to see quotes", options=options)

if aspect_pick:
    q_df = sub[sub["aspect"] == aspect_pick].copy()
    st.write(f"Matches: {len(q_df)} rows")
    # show a few distinct evidence strings
    shown = 0
    seen = set()
    for _, row in q_df.iterrows():
        ev = row.get("evidence") or ""
        if not ev or ev in seen:
            continue
        seen.add(ev)
        st.markdown(f"- **{(row.get('sentiment') or '').title()}** · "
                    f"Intensity: {row.get('intensity',1.0)} · "
                    f"Stars: {row.get('stars') if pd.notna(row.get('stars')) else '—'}  \n"
                    f"“{ev}”")
        shown += 1
        if shown >= 12:
            break
    if shown == 0:
        st.info("No distinct evidence quotes found for this aspect in current filters.")

# -----------------------------
# Download filtered data
# -----------------------------
st.subheader("Export")
dl_df = sub[["aspect","theme","sentiment","intensity","evidence","stars","overall_sentiment"]].copy()
st.download_button("Download filtered aspects (CSV)", data=dl_df.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_aspects.csv", mime="text/csv")

st.caption("Tip: Type a request like “negative durability 1–2 stars” or “positive fit 4–5 stars”, then refine with filters.")
