from google_play_scraper import reviews_all, Sort
import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts datetime objects to ISO format strings"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

review_options = [1, 2, 3, 4, 5]

for review_option in review_options:
    print(f"Processing review option {review_option}")
    result = reviews_all(
        'com.healthifyme.basic',
        sort=Sort.MOST_RELEVANT,
        lang='en',
        country='us',
        filter_score_with=review_option,
    )
    
    print(len(result))
    
    # Save into file
    with open(f'play_reviews/healthifyme_reviews_{review_option}.json', 'w') as f:
        json.dump(result, f, cls=DateTimeEncoder)
    
    print(f"Saved healthifyme_reviews_{review_option}.json")