# quick-commerce-insights
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load your dataset
# Expected columns: ['platform', 'review_text', 'sentiment'] 
# sentiment: ['positive', 'neutral', 'negative']
df = pd.read_csv('quick_commerce_reviews.csv')

# Drop missing values
df.dropna(subset=['platform', 'review_text', 'sentiment'], inplace=True)

# Encode target
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])  # 0=negative, 1=neutral, 2=positive

# Text vectorization
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(df['review_text'])
y = df['sentiment_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Add predictions to original DataFrame for ANOVA
df['predicted_sentiment'] = label_encoder.inverse_transform(model.predict(X))

# Sentiment numeric scale for ANOVA: negative=0, neutral=1, positive=2
df['sentiment_score'] = df['predicted_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# ANOVA test across platforms
anova_result = stats.f_oneway(
    df[df['platform'] == 'BlinkIt']['sentiment_score'],
    df[df['platform'] == 'Zepto']['sentiment_score'],
    df[df['platform'] == 'JioMart']['sentiment_score']
)

print("\nANOVA F-statistic:", anova_result.statistic)
print("ANOVA p-value:", anova_result.pvalue)

# Visualization
sns.boxplot(x='platform', y='sentiment_score', data=df)
plt.title("Customer Sentiment Comparison Across Platforms")
plt.ylabel("Sentiment Score (0=Neg, 1=Neu, 2=Pos)")
plt.xlabel("Platform")
plt.show()
