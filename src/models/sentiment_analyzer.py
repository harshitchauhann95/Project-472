"""
Sentiment Analysis Module
Analyzes the emotional tone / resident satisfaction level (positive, neutral, negative).
CRITICAL: As specified in the Smart Hostel report, sentiment is stored and reported
SEPARATELY from physical hazard, urgency, and priority scores.
"""

import re
from typing import Dict, Any


class SentimentAnalyzer:
    """
    Rule-based & lexicon-guided sentiment analyzer tailored for hostel maintenance context.
    Evaluates polarity, resident frustration, and polite appreciation.
    """
    
    POSITIVE_WORDS = {
        "thank", "thanks", "thankyou", "appreciate", "appreciated", "grateful",
        "resolved", "fixed", "better", "great", "excellent", "working", "good",
        "helpful", "prompt", "kindly", "pleased", "smooth"
    }
    
    NEGATIVE_WORDS = {
        "terrible", "horrible", "worst", "unacceptable", "furious", "angry", "annoyed",
        "frustrated", "repeatedly", "pathetic", "useless", "disgusting", "sick",
        "fed up", "again", "third time", "ignored", "awful", "suffering", "harassment",
        "unbearable", "dirty", "nightmare", "stinks", "ruined", "incompetent"
    }
    
    URGENCY_NEUTRAL_WORDS = {
        # Words indicating physical issues without emotional negativity
        "broken", "leak", "leaking", "sparking", "fused", "choked", "tripped", "loose", "cracked"
    }

    def __init__(self):
        pass

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Returns sentiment label ('positive', 'neutral', 'negative'), polarity score (-1.0 to 1.0),
        and reasoning cues.
        """
        if not text or not isinstance(text, str):
            return {"sentiment": "neutral", "polarity": 0.0, "cues": []}
            
        lower_t = text.lower()
        tokens = set(re.findall(r'\b\w+\b', lower_t))
        
        pos_hits = tokens.intersection(self.POSITIVE_WORDS)
        neg_hits = tokens.intersection(self.NEGATIVE_WORDS)
        
        # Check repetitive frustration patterns
        if "again" in lower_t or "repeated" in lower_t or "still not" in lower_t or "many times" in lower_t:
            neg_hits.add("repetition_frustration")
            
        pos_score = len(pos_hits)
        neg_score = len(neg_hits)
        
        if neg_score > pos_score:
            label = "negative"
            polarity = round(-min(1.0, 0.4 + 0.2 * neg_score), 2)
            cues = list(neg_hits)
        elif pos_score > neg_score:
            label = "positive"
            polarity = round(min(1.0, 0.4 + 0.2 * pos_score), 2)
            cues = list(pos_hits)
        else:
            label = "neutral"
            polarity = 0.0
            cues = ["factual_report"]
            
        return {
            "sentiment": label,
            "polarity": polarity,
            "cues": cues,
            "note": "Sentiment stored separately from physical priority score"
        }


# Quick test
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    samples = [
        "The ceiling fan is sparking near desk. Room 204.",
        "This is the third time I am reporting the washbasin leak! Disgusting service and totally unacceptable.",
        "Thank you team, the corridor light has been fixed, great prompt response!"
    ]
    for s in samples:
        print(s, "->", analyzer.analyze(s))
