import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mobile Phone Review Intelligence",
    page_icon="📱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

SENTIMENT_MODEL_PATH = MODEL_DIR / "best_sentiment_model.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
NMF_MODEL_PATH = MODEL_DIR / "nmf_topic_model.pkl"
ISSUE_VECTORIZER_PATH = MODEL_DIR / "issue_vectorizer.pkl"

PHONE_DATA_PATH = MODEL_DIR / "phone_recommendations.csv"
BRAND_DATA_PATH = MODEL_DIR / "brand_satisfaction.csv"
PROBLEM_PRODUCTS_PATH = MODEL_DIR / "problematic_products.csv"

# Optional file created by the extra notebook export cell below.
NEGATIVE_TOPICS_PATH = MODEL_DIR / "negative_review_topics.csv"


# ---------------------------------------------------------
# LOAD TRAINED ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    required_files = [
        SENTIMENT_MODEL_PATH,
        TFIDF_PATH,
        NMF_MODEL_PATH,
        ISSUE_VECTORIZER_PATH
    ]
    missing = [str(p.name) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing) +
            ". Run the model-saving cells in your Jupyter Notebook first."
        )

    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    nmf_model = joblib.load(NMF_MODEL_PATH)
    issue_vectorizer = joblib.load(ISSUE_VECTORIZER_PATH)

    return sentiment_model, tfidf_vectorizer, nmf_model, issue_vectorizer


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------
# TEXT PREPROCESSING
# Must match the preprocessing used in the notebook.
# ---------------------------------------------------------
@st.cache_resource
def get_nltk_tools():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Try to download missing NLTK resources.
    for resource in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK English stopwords are unavailable. Run "
            "`python -m nltk.downloader stopwords wordnet omw-1.4` "
            "in the same Python environment."
        ) from exc

    return stop_words, WordNetLemmatizer()


def clean_review(text, stop_words, lemmatizer):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # This mirrors the notebook's word_tokenize + lemmatization pipeline.
    # Regex tokenization is sufficient here because punctuation was removed.
    tokens = re.findall(r"[a-zA-Z]+", text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------
def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def money_label(value):
    if pd.isna(value):
        return "Price unavailable"
    return f"${value:,.2f}"


def topic_keywords(issue_vectorizer, nmf_model, n_words=10):
    terms = issue_vectorizer.get_feature_names_out()
    topics = []
    for topic_index, weights in enumerate(nmf_model.components_):
        top_indices = weights.argsort()[::-1][:n_words]
        topics.append({
            "Topic": f"Topic {topic_index + 1}",
            "Keywords": ", ".join(terms[i] for i in top_indices)
        })
    return pd.DataFrame(topics)


# ---------------------------------------------------------
# HEADER + SIDEBAR
# ---------------------------------------------------------
st.title("📱 Mobile Phone Recommendation & Review Intelligence")
st.caption(
    "Discover highly rated phones, analyse customer sentiment, "
    "and identify recurring product issues."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Phone Finder",
        "Review Sentiment Analyzer",
        "Common Customer Issues",
        "Brand Satisfaction",
        "Products Needing Attention"
    ]
)

try:
    sentiment_model, tfidf_vectorizer, nmf_model, issue_vectorizer = load_models()
    stop_words, lemmatizer = get_nltk_tools()
except Exception as error:
    st.error(f"Application setup issue: {error}")
    st.info(
        "Expected folder structure: app.py and a models/ folder containing "
        "the four .pkl files plus the CSV analysis exports."
    )
    st.stop()

phone_df = load_csv(str(PHONE_DATA_PATH))
brand_df = load_csv(str(BRAND_DATA_PATH))
problem_df = load_csv(str(PROBLEM_PRODUCTS_PATH))
topic_review_df = load_csv(str(NEGATIVE_TOPICS_PATH))

# Clean numeric columns if CSVs contain them as text.
for frame, columns in [
    (phone_df, ["Average_Price", "Average_Rating", "Total_Reviews",
                "Positive_Percentage", "Negative_Percentage",
                "Recommendation_Score"]),
    (brand_df, ["Average_Rating", "Total_Reviews",
                "Positive_Percentage", "Negative_Percentage"]),
    (problem_df, ["Average_Rating", "Total_Reviews", "Negative_Percentage"])
]:
    for column in columns:
        if column in frame.columns:
            frame[column] = safe_numeric(frame[column])


# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if page == "Project Overview":
    st.subheader("Project overview")
    st.write(
        "This application uses customer review text and star-rating analysis "
        "from the Amazon Unlocked Mobile dataset. It ranks products using "
        "review-derived measures and provides NLP-based review insights."
    )

    if not phone_df.empty:
        total_products = phone_df["Product Name"].nunique() if "Product Name" in phone_df else len(phone_df)
        total_reviews = phone_df["Total_Reviews"].sum() if "Total_Reviews" in phone_df else 0
        avg_rating = (
            np.average(
                phone_df["Average_Rating"].dropna(),
                weights=phone_df.loc[phone_df["Average_Rating"].notna(), "Total_Reviews"]
            )
            if {"Average_Rating", "Total_Reviews"}.issubset(phone_df.columns)
            and phone_df["Average_Rating"].notna().any()
            else np.nan
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Products analysed", f"{total_products:,.0f}")
        c2.metric("Reviews represented", f"{total_reviews:,.0f}")
        c3.metric("Weighted average rating", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "N/A")

        st.markdown("#### Top recommended phones")
        if "Recommendation_Score" in phone_df.columns:
            st.dataframe(
                phone_df.sort_values("Recommendation_Score", ascending=False).head(5),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("Phone recommendation CSV is missing. Run the notebook export cells.")

    st.markdown("#### What this application provides")
    st.markdown(
        "- **Phone Finder:** filter and rank phones by budget and review evidence.\n"
        "- **Review Sentiment Analyzer:** classify a new review as positive, neutral, or negative.\n"
        "- **Common Customer Issues:** inspect NMF topics discovered from negative reviews.\n"
        "- **Brand Satisfaction:** compare brand-level ratings and sentiment percentages.\n"
        "- **Products Needing Attention:** identify products with higher negative-review percentages."
    )

    st.warning(
        "Interpretation note: sentiment training labels were derived from star ratings "
        "(1–2 negative, 3 neutral, 4–5 positive). Model results therefore reproduce "
        "rating-derived labels and should be validated against manually labelled reviews "
        "before business-critical use."
    )


# ---------------------------------------------------------
# PAGE 2: PHONE FINDER
# ---------------------------------------------------------
elif page == "Phone Finder":
    st.subheader("🔎 Find a mobile phone")

    if phone_df.empty:
        st.error("models/phone_recommendations.csv was not found.")
        st.stop()

    required = {"Product Name", "Brand Name", "Total_Reviews", "Recommendation_Score"}
    if not required.issubset(phone_df.columns):
        st.error(f"Recommendation CSV must contain: {', '.join(sorted(required))}")
        st.stop()

    left, right = st.columns(2)

    with left:
        budget = st.number_input(
            "Maximum budget (USD)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            help="Enter 0 to show all prices."
        )

    with right:
        min_reviews = st.number_input(
            "Minimum number of reviews",
            min_value=1,
            value=20,
            step=5
        )

    # Curated allowlist: only recognized mobile-phone brands are shown.
    # This is an application-level quality filter, not an official certification.
    APPROVED_BRANDS = [
        "Acer", "Alcatel", "Apple", "Asus", "BlackBerry", "BLU",
        "CAT", "Coolpad", "Fairphone", "Google", "Honor", "HTC",
        "Huawei", "Infinix", "itel", "Kyocera", "Lava", "Lenovo",
        "LG", "Meizu", "Micromax", "Motorola", "Nokia", "Nothing",
        "OnePlus", "Oppo", "Panasonic", "Pantech", "POCO", "Realme",
        "Redmi", "Samsung", "Sharp", "Sony", "TCL", "Tecno",
        "Vertu", "Vivo", "Xiaomi", "Yota", "ZTE"
    ]

    # Match brand names without being affected by capitalization or spaces.
    approved_lookup = {brand.casefold(): brand for brand in APPROVED_BRANDS}
    phone_df = phone_df.copy()
    phone_df["Brand_Name_Canonical"] = (
        phone_df["Brand Name"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .map(approved_lookup)
    )

    # Exclude unknown, blank, and unrecognized brand labels from Phone Finder.
    recognized_phone_df = phone_df[
        phone_df["Brand_Name_Canonical"].notna()
    ].copy()

    available_brands = sorted(
        recognized_phone_df["Brand_Name_Canonical"].unique(),
        key=str.casefold
    )

    selected_brands = st.multiselect(
        "Filter by recognized mobile brand (leave empty for all recognized brands)",
        available_brands
    )

    result = recognized_phone_df.copy()
    result = result[result["Total_Reviews"] >= min_reviews]

    if budget > 0 and "Average_Price" in result.columns:
        result = result[
            result["Average_Price"].notna() &
            (result["Average_Price"] <= budget)
        ]

    if selected_brands:
        result = result[
            result["Brand_Name_Canonical"].isin(selected_brands)
        ]

    # Display the standardized brand name in the results table.
    result["Brand Name"] = result["Brand_Name_Canonical"]

    result = result.sort_values("Recommendation_Score", ascending=False)

    st.write(f"Found **{len(result):,}** matching products.")

    display_columns = [
        "Product Name", "Brand Name", "Average_Price", "Average_Rating",
        "Total_Reviews", "Positive_Percentage", "Negative_Percentage",
        "Recommendation_Score"
    ]
    display_columns = [c for c in display_columns if c in result.columns]

    st.dataframe(
        result[display_columns].head(100),
        use_container_width=True,
        hide_index=True
    )

    if not result.empty:
        st.markdown("#### Top recommendation")
        top = result.iloc[0]
        st.success(f"{top['Product Name']} — {top['Brand Name']}")
        a, b, c = st.columns(3)
        if "Average_Rating" in top:
            a.metric("Average rating", f"{top['Average_Rating']:.2f}/5")
        if "Total_Reviews" in top:
            b.metric("Review count", f"{int(top['Total_Reviews']):,}")
        if "Recommendation_Score" in top:
            c.metric("Recommendation score", f"{top['Recommendation_Score']:.3f}")

    st.caption(
        "Ranking is based on the notebook's review-derived score. It does not include "
        "hardware specifications such as processor, RAM, camera sensors, or battery capacity."
    )


# ---------------------------------------------------------
# PAGE 3: REVIEW SENTIMENT ANALYZER
# ---------------------------------------------------------
elif page == "Review Sentiment Analyzer":
    st.subheader("💬 Analyse a customer review")

    review_text = st.text_area(
        "Enter a mobile phone review",
        placeholder="Example: The battery drains quickly and the phone gets hot while charging.",
        height=150
    )

    if st.button("Analyse review", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = clean_review(review_text, stop_words, lemmatizer)

            if not cleaned.strip():
                st.warning(
                    "The review has no usable words after preprocessing. "
                    "Please enter a more descriptive review."
                )
            else:
                vector = tfidf_vectorizer.transform([cleaned])
                prediction = sentiment_model.predict(vector)[0]

                if str(prediction).lower() == "positive":
                    st.success(f"Predicted sentiment: {prediction}")
                elif str(prediction).lower() == "negative":
                    st.error(f"Predicted sentiment: {prediction}")
                else:
                    st.info(f"Predicted sentiment: {prediction}")

                st.write("**Cleaned review:**", cleaned)

                # A topic is most meaningful for negative reviews.
                if str(prediction).lower() == "negative":
                    issue_vector = issue_vectorizer.transform([cleaned])
                    topic_scores = nmf_model.transform(issue_vector)[0]
                    topic_index = int(np.argmax(topic_scores))

                    terms = issue_vectorizer.get_feature_names_out()
                    top_terms = nmf_model.components_[topic_index].argsort()[::-1][:10]
                    keywords = ", ".join(terms[i] for i in top_terms)

                    st.markdown("#### Related issue topic")
                    st.write(f"**Topic {topic_index + 1}**")
                    st.write(f"Top topic terms: {keywords}")
                    st.caption(
                        "This is a topic association, not a verified diagnosis of the complaint."
                    )

                st.caption(
                    "The model predicts sentiment learned from rating-derived labels. "
                    "It may make mistakes, especially for sarcasm, mixed opinions, or unusual wording."
                )


# ---------------------------------------------------------
# PAGE 4: COMMON CUSTOMER ISSUES
# ---------------------------------------------------------
elif page == "Common Customer Issues":
    st.subheader("🧩 Common customer issues")

    topic_table = topic_keywords(issue_vectorizer, nmf_model, n_words=10)
    st.write("NMF topics discovered from negative customer reviews:")
    st.dataframe(topic_table, use_container_width=True, hide_index=True)

    if not topic_review_df.empty and "Dominant_Topic" in topic_review_df.columns:
        topic_review_df["Dominant_Topic"] = safe_numeric(topic_review_df["Dominant_Topic"])
        counts = (
            topic_review_df["Dominant_Topic"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        plot_df = pd.DataFrame({
            "Topic": [f"Topic {i + 1}" for i in counts.index],
            "Negative review count": counts.values
        })

        st.markdown("#### Negative reviews by topic")
        st.bar_chart(plot_df.set_index("Topic"))

        st.markdown("#### Example customer complaints")
        topic_numbers = sorted(topic_review_df["Dominant_Topic"].dropna().astype(int).unique())
        selected_topic = st.selectbox(
            "Choose a topic",
            topic_numbers,
            format_func=lambda x: f"Topic {x + 1}"
        )

        sample_columns = [
            c for c in ["Brand Name", "Product Name", "Rating", "Reviews"]
            if c in topic_review_df.columns
        ]
        examples = topic_review_df[
            topic_review_df["Dominant_Topic"] == selected_topic
        ][sample_columns].head(10)

        st.dataframe(examples, use_container_width=True, hide_index=True)
    else:
        st.info(
            "For topic counts and example complaints, add the notebook export cell "
            "shown after this app code to create models/negative_review_topics.csv."
        )

    st.caption(
        "NMF topics are groups of co-occurring words. Review the example comments "
        "before assigning labels such as battery, camera, or heating."
    )


# ---------------------------------------------------------
# PAGE 5: BRAND SATISFACTION
# ---------------------------------------------------------
elif page == "Brand Satisfaction":
    st.subheader("🏷️ Brand customer satisfaction")

    if brand_df.empty:
        st.error("models/brand_satisfaction.csv was not found.")
        st.stop()

    if "Total_Reviews" in brand_df.columns:
        minimum_brand_reviews = st.slider(
            "Minimum brand review count",
            min_value=1,
            max_value=max(1, int(brand_df["Total_Reviews"].max())),
            value=min(20, max(1, int(brand_df["Total_Reviews"].max())))
        )
        view = brand_df[brand_df["Total_Reviews"] >= minimum_brand_reviews].copy()
    else:
        view = brand_df.copy()

    sort_column = "Positive_Percentage" if "Positive_Percentage" in view.columns else "Average_Rating"
    view = view.sort_values(sort_column, ascending=False)

    st.dataframe(view.head(100), use_container_width=True, hide_index=True)

    if {"Brand Name", "Positive_Percentage", "Negative_Percentage"}.issubset(view.columns):
        chart_data = view.nlargest(15, "Total_Reviews") if "Total_Reviews" in view else view.head(15)
        chart_data = chart_data.set_index("Brand Name")[
            ["Positive_Percentage", "Negative_Percentage"]
        ]
        st.markdown("#### Positive vs negative review percentage")
        st.bar_chart(chart_data)

    st.caption(
        "Brand comparisons should use a minimum review threshold because brands "
        "with very few reviews can have unstable percentages."
    )


# ---------------------------------------------------------
# PAGE 6: PRODUCTS NEEDING ATTENTION
# ---------------------------------------------------------
elif page == "Products Needing Attention":
    st.subheader("⚠️ Products needing attention")

    if problem_df.empty:
        st.error("models/problematic_products.csv was not found.")
        st.stop()

    if "Negative_Percentage" in problem_df.columns:
        problem_df = problem_df.sort_values("Negative_Percentage", ascending=False)

    st.dataframe(
        problem_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    if {"Product Name", "Negative_Percentage"}.issubset(problem_df.columns):
        chart_data = problem_df.head(15).copy()
        chart_data["Product"] = chart_data["Product Name"].astype(str).str.slice(0, 45)
        st.markdown("#### Products with highest negative-review percentage")
        st.bar_chart(chart_data.set_index("Product")["Negative_Percentage"])

    st.info(
        "A high negative-review percentage is a signal for investigation, not proof "
        "of a product defect. Review complaint examples and account for review volume."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption("Mobile Phone Recommendation & Customer Review Intelligence | Built with Python, Scikit-learn, NLTK and Streamlit")
