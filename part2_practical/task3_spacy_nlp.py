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
import os

print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
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
                     'reliable', 'beautiful', 'perfectly', 'much better', 'works perfectly'}
    negative_words = {'damaged', 'disappointed', 'terrible', 'poor', 'broken',
                     'disappointing', 'bad', 'awful', 'horrible', 'could be faster'}
    
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
colors = ['green', 'red', 'gray']
plt.bar(sentiments, sentiment_values, color=colors[:len(sentiments)])
plt.title('Sentiment Distribution')
plt.ylabel('Number of Reviews')

plt.tight_layout()
plt.savefig('nlp_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a function to save NER visualization properly
def save_ner_visualization(text, filename):
    """Save NER visualization as HTML"""
    doc = nlp(text)
    
    # Generate HTML with proper styling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NER Visualization</title>
        <style>
            .entity {{ 
                padding: 2px 4px;
                border-radius: 3px;
                font-weight: bold;
            }}
            .ORG {{ background: #7aecec; }}
            .PRODUCT {{ background: #feca74; }}
            .GPE {{ background: #ff9561; }}
            .CARDINAL {{ background: #e4e7d2; }}
            .DATE {{ background: #bfe1d9; }}
        </style>
    </head>
    <body>
        <h2>Named Entity Recognition Visualization</h2>
        <p><strong>Text:</strong> {text}</p>
        <div style="border: 1px solid #ccc; padding: 15px; margin: 10px 0;">
    """
    
    # Add entities with styling
    for ent in doc.ents:
        html += f'<mark class="entity {ent.label_}" title="{ent.label_}">{ent.text}</mark> '
    
    html += """
        </div>
        <h3>Entity Legend:</h3>
        <ul>
            <li><span style="background: #7aecec; padding: 2px 5px;">ORG</span> - Organization</li>
            <li><span style="background: #feca74; padding: 2px 5px;">PRODUCT</span> - Product</li>
            <li><span style="background: #ff9561; padding: 2px 5px;">GPE</span> - Geo-political Entity</li>
            <li><span style="background: #e4e7d2; padding: 2px 5px;">CARDINAL</span> - Cardinal Number</li>
            <li><span style="background: #bfe1d9; padding: 2px 5px;">DATE</span> - Date</li>
        </ul>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"NER visualization saved as {filename}")

# Save NER visualization for a sample review
print("\nCreating NER visualization for sample review...")
sample_review = reviews[0]
save_ner_visualization(sample_review, 'ner_visualization.html')

# Alternative: Create a text-based NER display
def display_ner_text(text):
    """Display NER results in text format"""
    doc = nlp(text)
    print(f"\nText: {text}")
    print("Entities found:")
    for ent in doc.ents:
        print(f"  - {ent.text} ({ent.label_})")

print("\n" + "="*50)
print("TEXT-BASED NER DISPLAY")
print("="*50)
display_ner_text(sample_review)

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

# Enhanced sentiment analysis by brand
print("\n" + "="*50)
print("SENTIMENT ANALYSIS BY BRAND")
print("="*50)

major_brands = ['Apple', 'Samsung', 'Sony', 'Google', 'Microsoft', 'Dell', 'Bose', 'Lenovo', 'HP']

for brand in major_brands:
    brand_reviews = [r for r in results if brand.lower() in r['review'].lower()]
    if brand_reviews:
        sentiments = [r['sentiment'] for r in brand_reviews]
        sentiment_dist = Counter(sentiments)
        total_reviews = len(brand_reviews)
        
        print(f"\n{brand} (mentioned in {total_reviews} review{'s' if total_reviews > 1 else ''}):")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_dist.get(sentiment, 0)
            percentage = (count / total_reviews) * 100 if total_reviews > 0 else 0
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")

# Detailed analysis of sentiment patterns
print("\n" + "="*50)
print("SENTIMENT PATTERN ANALYSIS")
print("="*50)

positive_reviews = [r for r in results if r['sentiment'] == 'positive']
negative_reviews = [r for r in results if r['sentiment'] == 'negative']

print(f"\nPositive Reviews ({len(positive_reviews)}):")
for review in positive_reviews:
    print(f"  - {review['review'][:80]}...")

print(f"\nNegative Reviews ({len(negative_reviews)}):")
for review in negative_reviews:
    print(f"  - {review['review'][:80]}...")

# Create a summary report
print("\n" + "="*60)
print("FINAL SUMMARY REPORT")
print("="*60)

total_reviews = len(results)
positive_count = len([r for r in results if r['sentiment'] == 'positive'])
negative_count = len([r for r in results if r['sentiment'] == 'negative'])
neutral_count = len([r for r in results if r['sentiment'] == 'neutral'])

print(f"\nOverall Sentiment Distribution:")
print(f"  Positive: {positive_count}/{total_reviews} ({positive_count/total_reviews*100:.1f}%)")
print(f"  Negative: {negative_count}/{total_reviews} ({negative_count/total_reviews*100:.1f}%)")
print(f"  Neutral:  {neutral_count}/{total_reviews} ({neutral_count/total_reviews*100:.1f}%)")

print(f"\nTotal Entities Extracted: {len(all_entities)}")
print(f"Unique Entity Types: {len(entity_types)}")
print(f"Most Common Entity Type: {entity_types.most_common(1)[0][0]}")

print(f"\n✅ Task 3 completed successfully!")
print(f"📊 Analyzed {len(reviews)} reviews with spaCy")
print(f"📈 Generated visualizations: nlp_analysis.png, ner_visualization.html")
print(f"📋 Created comprehensive sentiment and entity analysis")
