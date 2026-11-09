# Run this after the NMF topic assignment cell (after negative_reviews['Dominant_Topic'] is created).
# This saves review-level topic assignments so Streamlit can show topic counts and example complaints.

negative_reviews.to_csv(
    'models/negative_review_topics.csv',
    index=False
)

print("Negative review topic data saved successfully!")
