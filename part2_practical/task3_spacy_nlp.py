"""
Task 3: NLP with spaCy - NER and Sentiment Analysis
Text Data: Amazon Product Reviews
Goal: Perform NER and rule-based sentiment analysis
"""

import spacy
from spacy import displacy
import random
from collections import Counter
import matplotlib.pyplot as plt

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews
reviews = [
    "I bought the new iPhone 15 from Apple and it's absolutely amazing! The camera quality is outstanding.",
    "Samsung Galaxy S23 Ultra arrived damaged. Very disappointed with the shipping from Amazon.",
    "The Sony WH-1000XM4 headphones have incredible noise cancellation. Best purchase ever!",
    "Google Pixel 7 Pro has terrible battery life. Would not recommend to anyone.",
    "Microsoft Surface Laptop 5 works perfectly for my programming tasks. Great performance with Intel i7 processor.",
    "The Dell XPS 13 is okay, but the MacBook Air M2 from Apple is much better for creative work.",
    "Bose QuietComfort headphones broke after 2 months. Poor quality for the price.",
    "Lenovo ThinkPad is reliable as always. Perfect for business use with Windows 11.",
    "The new iPad Pro from Apple with M2 chip is revolutionary for digital artists.",
    "HP Spectre x360 has beautiful design but the AMD processor could be faster."
]

print(f"Analyzing {len(reviews)} product reviews...")

# Perform NER and sentiment analysis
def analyze_sentiment(text):
    """Rule-based sentiment analysis"""
    positive_words = {'amazing', 'outstanding', 'incredible', 'best', 'great', 
                     'perfect', 'excellent', 'awesome', 'good', 'revolutionary',
                     'reliable', 'beautiful', 'perfectly'}
    negative_words = {'damaged', 'disappointed', 'terrible', 'poor', 'broken',
                     'disappointing', 'bad', 'awful', 'horrible'}
    
    doc = nlp(text.lower())
    positive_count = sum(1 for token in doc if token.text in positive_words)
    negative_count = sum(1 for token in doc if token.text in negative_words)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# Analyze each review
results = []
all_entities = []

print("\n" + "="*60)
print("REVIEW ANALYSIS RESULTS")
print("="*60)

for i, review in enumerate(reviews, 1):
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = analyze_sentiment(review)
    
    results.append({
        'review': review,
        'entities': entities,
        'sentiment': sentiment
    })
    
    all_entities.extend(entities)
    
    print(f"\nReview {i}:")
    print(f"Text: {review}")
    print(f"Sentiment: {sentiment.upper()}")
    print(f"Entities found: {entities}")

# Entity statistics
entity_types = Counter([label for _, label in all_entities])
entities_text = Counter([text for text, _ in all_entities])

print(f"\n{'='*50}")
print("ENTITY ANALYSIS SUMMARY")
print(f"{'='*50}")

print(f"\nTotal entities found: {len(all_entities)}")
print("\nEntity types distribution:")
for entity_type, count in entity_types.most_common():
    print(f"{entity_type}: {count}")

print("\nMost common entities:")
for entity, count in entities_text.most_common(10):
    print(f"'{entity}': {count}")

# Visualize entity distribution
plt.figure(figsize=(12, 5))

# Entity types pie chart
plt.subplot(1, 2, 1)
labels, counts = zip(*entity_types.most_common())
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Entity Types')

# Sentiment distribution
plt.subplot(1, 2, 2)
sentiment_counts = Counter([r['sentiment'] for r in results])
sentiments, sentiment_values = zip(*sentiment_counts.items())
plt.bar(sentiments, sentiment_values, color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.ylabel('Number of Reviews')

plt.tight_layout()
plt.savefig('nlp_analysis.png')
plt.show()

# Visualize NER for a sample review
print("\nVisualizing NER for sample review...")
sample_review = reviews[0]
doc = nlp(sample_review)

# Display NER visualization
displacy.render(doc, style='ent', jupyter=False)

# Save NER visualization
html = displacy.render(doc, style='ent', page=True)
with open('ner_visualization.html', 'w', encoding='utf-8') as f:
    f.write(html)

# Product and brand analysis
print(f"\n{'='*50}")
print("PRODUCT AND BRAND ANALYSIS")
print(f"{'='*50}")

# Extract products and brands
products_brands = []
for ent in all_entities:
    if ent[1] in ['PRODUCT', 'ORG']:
        products_brands.append(ent[0])

brand_product_counts = Counter(products_brands)
print("\nMost mentioned products/brands:")
for item, count in brand_product_counts.most_common():
    print(f"{item}: {count} mentions")

# Sentiment by brand (simplified)
print("\nSentiment analysis by major brands:")
major_brands = ['Apple', 'Samsung', 'Sony', 'Google', 'Microsoft', 'Dell']

for brand in major_brands:
    brand_reviews = [r for r in results if brand.lower() in r['review'].lower()]
    if brand_reviews:
        sentiments = [r['sentiment'] for r in brand_reviews]
        sentiment_dist = Counter(sentiments)
        print(f"\n{brand}:")
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} reviews")

print(f"\n✅ Task 3 completed! Analyzed {len(reviews)} reviews with spaCy.")
