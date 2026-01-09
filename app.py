import streamlit as st 
import joblib
import numpy as np

model= joblib.load('models/sentiment_model.pkl')
vectorizer= joblib.load('models/tfidf_vectorizer.pkl')

st.set_page_config(page_title="SmartDine AI",page_icon="🍽️",layout="wide")

st.title("🍽️ SmartDine AI- Restaurant Sentiment Analysis")

st.write("Enter a restaurant review to see how sentiment changes in real time based on the text.")

review= st.text_area("Restaurent review")

if st.button("Analyze Sentiment"):
    if review.strip():
        review_vector= vectorizer.transform([review])
        prediction= model.predict(review_vector)[0]
        probability= model.predict_proba(review_vector)[0]

        confidence= np.max(probability)*100

        if prediction==1:
            st.success(f"Positive Review 😁 \n Confidence:{confidence:.2f}%")
        else:
            st.error(f"Negative Review 😖 \n Confidence:{confidence:.2f}%")

        #show influential words 
        st.subheader("Words influencing this prediction")

        feature_names= vectorizer.get_feature_names_out()
        coefs= model.coef_[0]

        word_scores= review_vector.toarray()[0]* coefs
        top_indices= np.argsort(np.abs(word_scores))[-5:]

       # Change your loop to this for better readability
    for i in reversed(top_indices):
      if word_scores[i] != 0: # Only show words that were actually in the review
        st.write(f"- **{feature_names[i]}**: {word_scores[i]:.4f}")

        
        st.caption("Small chnages in words can shift predictions, showing how sensitive text models are to language")
