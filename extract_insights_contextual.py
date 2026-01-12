import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import the base categorization from the explainable version
import sys
from extract_insights_explainable import (
    categorize_review, BUG_PATTERNS, FEATURE_REQUEST_PATTERNS,
    UX_ISSUE_PATTERNS, PERFORMANCE_PATTERNS, PRAISE_PATTERNS,
    PAIN_POINT_PATTERNS, ADS_PATTERNS, COMPETITOR_PATTERNS
)

# ============================================================================
# CONTEXTUAL INSIGHT EXTRACTION
# ============================================================================

def extract_contextual_insights(df, category, top_n=20):
    """Extract representative examples with context for a category"""
    
    # Filter reviews that match this category
    matching_reviews = df[df[category].apply(lambda x: len(x) > 0 if x else False)].copy()
    
    if len(matching_reviews) == 0:
        return []
    
    # Sort by relevance: number of matches * thumbs up count
    matching_reviews['relevance'] = (
        matching_reviews[category].apply(len) * 
        (matching_reviews['thumbsUpCount'] + 1)  # +1 to avoid zeros
    )
    matching_reviews = matching_reviews.sort_values('relevance', ascending=False)
    
    insights = []
    for _, row in matching_reviews.head(top_n).iterrows():
        insights.append({
            'content': row['content'],
            'score': row['score'],
            'thumbsUp': row['thumbsUpCount'],
            'matched_phrases': row[category],
            'reviewId': row['reviewId']
        })
    
    return insights

def cluster_similar_issues(df, category):
    """Group reviews by similar matched phrases"""
    clusters = defaultdict(list)
    
    for _, row in df.iterrows():
        if row[category] and len(row[category]) > 0:
            # Use the first matched phrase as cluster key
            key_phrase = row[category][0] if isinstance(row[category], list) else str(row[category])
            clusters[key_phrase].append({
                'content': row['content'],
                'score': row['score'],
                'thumbsUp': row['thumbsUpCount'],
                'matched': row[category]
            })
    
    # Sort clusters by impact (count * avg thumbs up)
    cluster_summary = []
    for phrase, reviews in clusters.items():
        total_thumbs = sum(r['thumbsUp'] for r in reviews)
        avg_rating = sum(r['score'] for r in reviews) / len(reviews)
        
        cluster_summary.append({
            'key_phrase': phrase,
            'count': len(reviews),
            'total_impact': total_thumbs + len(reviews),
            'avg_rating': avg_rating,
            'examples': reviews[:5]  # Top 5 examples
        })
    
    cluster_summary.sort(key=lambda x: x['total_impact'], reverse=True)
    return cluster_summary

def extract_specific_requests(df):
    """Extract specific feature requests with context using more targeted patterns"""
    feature_reviews = df[df['feature_requests'].apply(lambda x: len(x) > 0 if x else False)]
    
    specific_requests = []
    
    # Common feature request patterns
    request_patterns = [
        (r'(add|include|need|want|wish).{0,50}?(feature|option|ability|function)', 'Feature Addition'),
        (r'(integrate|integration|sync|connect).{0,30}?(with|to)', 'Integration Request'),
        (r'(customize|custom|personalize).{0,30}', 'Customization'),
        (r'(export|download|backup).{0,30}', 'Data Export'),
        (r'(dark mode|theme|color scheme)', 'UI Theme'),
        (r'(widget|notification|reminder)', 'App Features'),
    ]
    
    for _, row in feature_reviews.iterrows():
        content = row['content'].lower()
        for pattern, category_name in request_patterns:
            if re.search(pattern, content):
                specific_requests.append({
                    'category': category_name,
                    'content': row['content'],
                    'score': row['score'],
                    'thumbsUp': row['thumbsUpCount'],
                    'matched': row['feature_requests']
                })
                break  # Only categorize once
    
    return specific_requests

# ============================================================================
# ENHANCED REPORTING
# ============================================================================

def generate_contextual_report(df):
    """Generate a PM-friendly report with actionable insights and context"""
    lines = []
    lines.append("=" * 100)
    lines.append("ACTIONABLE PRODUCT INSIGHTS - WITH CONTEXT")
    lines.append("=" * 100)
    lines.append(f"\nTotal reviews analyzed: {len(df)}")
    lines.append(f"Date range: {df['at'].min()} to {df['at'].max()}\n")
    
    # Categories to analyze
    categories = {
        'bugs': ('CRITICAL BUGS & CRASHES', BUG_PATTERNS),
        'feature_requests': ('FEATURE REQUESTS', FEATURE_REQUEST_PATTERNS),
        'performance': ('PERFORMANCE ISSUES', PERFORMANCE_PATTERNS),
        'ux_issues': ('UX/USABILITY ISSUES', UX_ISSUE_PATTERNS),
        'ads_complaints': ('ADS & MONETIZATION COMPLAINTS', ADS_PATTERNS),
        'pain_points': ('MAJOR PAIN POINTS', PAIN_POINT_PATTERNS),
    }
    
    for cat_key, (cat_title, patterns) in categories.items():
        lines.append("\n" + "=" * 100)
        lines.append(f"{cat_title}")
        lines.append("=" * 100)
        
        # Get clusters
        clusters = cluster_similar_issues(df, cat_key)
        
        if not clusters:
            lines.append("\n✓ No significant issues found in this category.\n")
            continue
        
        lines.append(f"\nTotal reviews mentioning this: {sum(c['count'] for c in clusters)}")
        lines.append(f"Unique issue types: {len(clusters)}\n")
        
        # Show top clustered issues
        for i, cluster in enumerate(clusters[:10], 1):  # Top 10 clusters
            lines.append(f"\n{i}. ISSUE: '{cluster['key_phrase']}'")
            lines.append(f"   Frequency: {cluster['count']} reviews | Avg Rating: {cluster['avg_rating']:.1f}★")
            lines.append(f"   Impact Score: {cluster['total_impact']}")
            lines.append(f"\n   Example reviews:")
            
            for j, example in enumerate(cluster['examples'][:3], 1):  # Top 3 examples
                # Truncate long reviews
                content_preview = example['content'][:200] + "..." if len(example['content']) > 200 else example['content']
                lines.append(f"\n   [{j}] ({example['score']}★, {example['thumbsUp']} 👍)")
                lines.append(f"       \"{content_preview}\"")
            
            lines.append("")  # Spacing
    
    # Specific Feature Requests Section
    lines.append("\n" + "=" * 100)
    lines.append("SPECIFIC FEATURE REQUESTS - CATEGORIZED")
    lines.append("=" * 100)
    
    specific = extract_specific_requests(df)
    if specific:
        by_category = defaultdict(list)
        for req in specific:
            by_category[req['category']].append(req)
        
        for cat_name, requests in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"\n{cat_name.upper()} ({len(requests)} requests)")
            lines.append("-" * 80)
            
            # Show top examples
            sorted_reqs = sorted(requests, key=lambda x: x['thumbsUp'], reverse=True)[:5]
            for req in sorted_reqs:
                content_preview = req['content'][:150] + "..." if len(req['content']) > 150 else req['content']
                lines.append(f"\n  • ({req['score']}★, {req['thumbsUp']} 👍) \"{content_preview}\"")
    else:
        lines.append("\nNo specific feature requests extracted.")
    
    # Competitor Analysis
    lines.append("\n" + "=" * 100)
    lines.append("COMPETITOR MENTIONS")
    lines.append("=" * 100)
    
    competitor_reviews = df[df['competitor_mentions'].apply(lambda x: len(x) > 0 if x else False)]
    if len(competitor_reviews) > 0:
        lines.append(f"\n{len(competitor_reviews)} reviews mention competitors\n")
        
        for _, row in competitor_reviews.head(10).iterrows():
            content_preview = row['content'][:200] + "..." if len(row['content']) > 200 else row['content']
            lines.append(f"  • Mentioned: {', '.join(row['competitor_mentions'])}")
            lines.append(f"    ({row['score']}★, {row['thumbsUpCount']} 👍) \"{content_preview}\"\n")
    else:
        lines.append("\nNo competitor mentions found.")
    
    # Praise & What's Working
    lines.append("\n" + "=" * 100)
    lines.append("WHAT'S WORKING WELL (PRAISE)")
    lines.append("=" * 100)
    
    praise_clusters = cluster_similar_issues(df, 'praise')[:5]  # Top 5
    if praise_clusters:
        for i, cluster in enumerate(praise_clusters, 1):
            lines.append(f"\n{i}. Users love: '{cluster['key_phrase']}'")
            lines.append(f"   Mentioned in {cluster['count']} reviews")
            
            for example in cluster['examples'][:2]:
                content_preview = example['content'][:150] + "..." if len(example['content']) > 150 else example['content']
                lines.append(f"   • \"{content_preview}\"")
    
    return '\n'.join(lines)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting Contextual Insights Extraction...")
    print("=" * 80)
    
    # Load the explainable analysis results
    print("\nLoading previous analysis...")
    df = pd.read_json('analysis/pm_insights_explainable.json')
    
    print(f"Loaded {len(df)} analyzed reviews")
    
    # Generate contextual report
    print("\nGenerating contextual insights report...")
    contextual_report = generate_contextual_report(df)
    
    # Save report
    output_file = 'analysis/pm_insights_contextual.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(contextual_report)
    
    print(f"\n✓ Contextual report generated!")
    print(f"  Output: {output_file}")
    print("\nThis report includes:")
    print("  • Clustered issues with example reviews")
    print("  • Specific feature requests categorized")
    print("  • Competitor mentions with context")
    print("  • What users love (to preserve)")
    print("\nNow you have the CONTEXT you need for actionable decisions!")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("PREVIEW:")
    print("=" * 80)
    print(contextual_report[:2000])  # First 2000 chars
    print("\n... (see full report in analysis/pm_insights_contextual.txt)")
