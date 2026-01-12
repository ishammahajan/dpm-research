import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load spaCy model for NER and lemmatization
print("Loading NLP components...")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
    print("✓ SpaCy loaded with NER support")
except (ImportError, OSError) as e:
    USE_SPACY = False
    print(f"⚠ SpaCy not available: {e}")
    print("  Using regex-based extraction (still works!)")

# ============================================================================
# PATTERN DEFINITIONS - Fully Explainable Rules
# ============================================================================

BUG_PATTERNS = [
    r'\b(crash|crashes|crashed|crashing)\b',
    r'\b(freeze|freezes|frozen|freezing)\b',
    r'\b(error|errors|bug|bugs|glitch|glitches)\b',
    r'\b(not work|doesn\'t work|stopped working|won\'t work)\b',
    r'\b(broken|breaks|broke)\b',
    r'\b(stuck|hanging|hangs)\b',
    r'\b(problem|issue|issues)\b.*\b(technical|loading|saving)\b',
]

FEATURE_REQUEST_PATTERNS = [
    r'\b(wish|hope|would like|want|need)\b.*\b(feature|option|ability|function)\b',
    r'\b(please add|should add|could add|add)\b',
    r'\b(missing|lacks|doesn\'t have)\b',
    r'\b(would be (great|nice|good|better))\b.*\bif\b',
    r'\b(suggestion|suggest|recommend)\b',
]

UX_ISSUE_PATTERNS = [
    r'\b(confusing|confused|hard to (use|find|navigate))\b',
    r'\b(difficult|complicated|complex)\b.*\b(interface|ui|ux|menu)\b',
    r'\b(annoying|frustrating|irritating)\b',
    r'\b(clunky|awkward|unintuitive)\b',
    r'\b(too many (clicks|steps|taps))\b',
    r'\b(hard to understand|can\'t find)\b',
]

PERFORMANCE_PATTERNS = [
    r'\b(slow|laggy|lags|lagging|sluggish)\b',
    r'\b(takes (too long|forever))\b',
    r'\b(battery drain|drains battery)\b',
    r'\b(loading|(load|loads) (slowly|slow))\b',
    r'\b(performance|speed)\b.*\b(bad|poor|terrible|slow)\b',
]

PRAISE_PATTERNS = [
    r'\b(love|loves|loved|loving)\b',
    r'\b(great|amazing|awesome|excellent|fantastic|wonderful)\b',
    r'\b(perfect|best|favorite|favourite)\b',
    r'\b(helpful|useful|convenient)\b',
    r'\b(easy|simple|straightforward)\b.*\b(use|interface)\b',
]

PAIN_POINT_PATTERNS = [
    r'\b(hate|annoying|frustrating|disappointed)\b',
    r'\b(waste of (time|money))\b',
    r'\b(regret|mistake)\b.*\b(download|install|purchase|buying)\b',
    r'\b(terrible|horrible|awful|worst)\b',
    r'\b(uninstall|delete|removing)\b',
]

ADS_PATTERNS = [
    r'\b(too many ads|ads everywhere|ad spam)\b',
    r'\b(ads? (are|is)).*\b(annoying|excessive|intrusive|overwhelming)\b',
    r'\b(constant ads|nonstop ads|non-stop ads)\b',
    r'\b(can\'t.*without.*ad|force.*watch.*ad)\b',
]

COMPETITOR_PATTERNS = [
    r'\b(my ?fitness ?pal|myfitnesspal|mfp)\b',
    r'\b(lose ?it|loseit)\b',
    r'\b(noom)\b',
    r'\b(calorie ?counter|fat ?secret)\b',
    r'\b(fitness ?pal)\b',
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def match_patterns(text, patterns):
    """Match patterns and return matched phrases for explainability"""
    if not text:  # Handle None or empty string
        return []
    text_lower = text.lower()
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text_lower, re.IGNORECASE)
        if found:
            matches.extend(found if isinstance(found[0], str) else [f[0] if isinstance(f, tuple) else f for f in found])
    return list(set(matches))  # Return unique matches

def extract_named_entities(text):
    """Extract named entities (apps, products, features)"""
    if not text:  # Handle None or empty string
        return {'products': [], 'features': []}
    
    if USE_SPACY:
        doc = nlp(text)
        entities = {
            'products': [ent.text for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG']],
            'features': []  # Could be enhanced with custom rules
        }
    else:
        # Fallback: simple extraction
        entities = {
            'products': [],
            'features': []
        }
    return entities

def categorize_review(review_text):
    """Categorize a review using pattern matching"""
    categories = {
        'bugs': match_patterns(review_text, BUG_PATTERNS),
        'feature_requests': match_patterns(review_text, FEATURE_REQUEST_PATTERNS),
        'ux_issues': match_patterns(review_text, UX_ISSUE_PATTERNS),
        'performance': match_patterns(review_text, PERFORMANCE_PATTERNS),
        'praise': match_patterns(review_text, PRAISE_PATTERNS),
        'pain_points': match_patterns(review_text, PAIN_POINT_PATTERNS),
        'ads_complaints': match_patterns(review_text, ADS_PATTERNS),
        'competitor_mentions': match_patterns(review_text, COMPETITOR_PATTERNS),
    }
    
    entities = extract_named_entities(review_text)
    
    return {
        **categories,
        'entities': entities,
        'has_insights': any(len(v) > 0 for v in categories.values())
    }

# ============================================================================
# TOPIC MODELING
# ============================================================================

def perform_topic_modeling(reviews, n_topics=10, n_top_words=10):
    """Discover topics using LDA"""
    print(f"\nPerforming topic modeling with {n_topics} topics...")
    
    # Prepare text - filter out None or empty content
    texts = [r['content'] for r in reviews if r.get('content')]
    
    # Vectorize
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=5,  # Must appear in at least 5 documents
        max_df=0.7  # Must appear in less than 70% of documents
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20
    )
    lda.fit(doc_term_matrix)
    
    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_id': topic_idx,
            'top_words': top_words,
            'theme': ', '.join(top_words[:5])  # First 5 words as theme
        })
    
    return topics, lda, vectorizer

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_reviews(json_file, sample_size=None):
    """Process reviews from a JSON file"""
    print(f"\nProcessing {json_file}...")
    
    # Read JSON array from file
    with open(json_file, 'r') as f:
        reviews = json.load(f)
    
    if sample_size:
        reviews = reviews[:sample_size]
        print(f"Analyzing {sample_size} sample reviews...")
    else:
        print(f"Analyzing {len(reviews)} reviews...")
    
    insights = []
    
    for review in tqdm(reviews, desc="Categorizing reviews"):
        categorization = categorize_review(review['content'])
        
        insights.append({
            'reviewId': review['reviewId'],
            'userName': review['userName'],
            'score': review['score'],
            'content': review['content'],
            'thumbsUpCount': review['thumbsUpCount'],
            'at': review['at'],
            **{k: categorization[k] for k in categorization if k != 'entities'},
            'mentioned_products': categorization['entities']['products'],
        })
    
    return insights, reviews

# ============================================================================
# REPORTING
# ============================================================================

def generate_summary_report(df, topics):
    """Generate a comprehensive summary report"""
    lines = []
    lines.append("=" * 80)
    lines.append("PRODUCT INSIGHTS SUMMARY REPORT")
    lines.append("Explainable NLP Analysis - Pattern Matching + Topic Modeling")
    lines.append("=" * 80)
    lines.append(f"\nTotal reviews analyzed: {len(df)}")
    lines.append(f"Reviews with actionable insights: {df['has_insights'].sum()}")
    lines.append(f"Percentage with insights: {df['has_insights'].mean()*100:.1f}%\n")
    
    # Category statistics
    categories = ['bugs', 'feature_requests', 'ux_issues', 'performance', 
                  'praise', 'pain_points', 'ads_complaints', 'competitor_mentions']
    
    lines.append("\nINSIGHT CATEGORIES")
    lines.append("-" * 80)
    for cat in categories:
        count = sum(len(x) > 0 for x in df[cat] if x)
        if count > 0:
            lines.append(f"\n{cat.upper().replace('_', ' ')}: {count} reviews")
            
            # Get top patterns matched
            all_matches = []
            for matches in df[cat]:
                if matches:
                    all_matches.extend(matches)
            
            if all_matches:
                top_matches = Counter(all_matches).most_common(10)
                for phrase, freq in top_matches:
                    lines.append(f"  {freq:4d}x  '{phrase}'")
    
    # Topic analysis
    lines.append("\n\nDISCOVERED TOPICS (LDA Topic Modeling)")
    lines.append("-" * 80)
    for topic in topics:
        lines.append(f"\nTopic {topic['topic_id'] + 1}: {topic['theme']}")
        lines.append(f"  Keywords: {', '.join(topic['top_words'])}")
    
    # Star rating correlation
    lines.append("\n\nCATEGORY DISTRIBUTION BY STAR RATING")
    lines.append("-" * 80)
    for cat in categories:
        df[f'{cat}_count'] = df[cat].apply(lambda x: len(x) if x else 0)
        df[f'{cat}_present'] = df[f'{cat}_count'] > 0
        
        by_rating = df.groupby('score')[f'{cat}_present'].sum()
        if by_rating.sum() > 0:
            lines.append(f"\n{cat.replace('_', ' ').title()}:")
            for rating in sorted(by_rating.index):
                lines.append(f"  {rating} star: {by_rating[rating]:4d} reviews")
    
    return '\n'.join(lines)

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting Explainable Product Insights Extraction")
    print("=" * 80)
    
    # Configuration
    SAMPLE_SIZE = None  # Set to number for testing, None for all reviews
    N_TOPICS = 15  # Number of topics to discover
    
    # Process all review files
    all_insights = []
    all_reviews = []
    review_files = sorted(Path('play_reviews').glob('healthifyme_reviews_*.json'))
    
    for review_file in review_files:
        insights, reviews = process_reviews(review_file, SAMPLE_SIZE)
        all_insights.extend(insights)
        all_reviews.extend(reviews)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_insights)
    
    # Perform topic modeling on all reviews
    topics, lda_model, vectorizer = perform_topic_modeling(all_reviews, N_TOPICS)
    
    # Save detailed results
    print("\nSaving results...")
    
    # CSV with categorization
    df.to_csv('analysis/pm_insights_explainable.csv', index=False)
    
    # JSON with full details
    df.to_json('analysis/pm_insights_explainable.json', orient='records', indent=2)
    
    # Topics as JSON
    with open('analysis/discovered_topics.json', 'w') as f:
        json.dump(topics, f, indent=2)
    
    # Generate and save summary report
    summary = generate_summary_report(df, topics)
    print(summary)
    
    with open('analysis/pm_insights_summary.txt', 'w') as f:
        f.write(summary)
    
    print(f"\n\n✓ Analysis complete!")
    print(f"  Detailed results: analysis/pm_insights_explainable.csv / .json")
    print(f"  Discovered topics: analysis/discovered_topics.json")
    print(f"  Summary report: analysis/pm_insights_summary.txt")
    print(f"\nAll insights are fully explainable - check CSV for matched patterns!")
