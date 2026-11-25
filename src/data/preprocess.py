"""
Preprocessing and Cleaning Pipeline
Maps multi-parameter building maintenance complaints into the 5 report categories:
  1. electrical
  2. plumbing
  3. furniture_carpentry
  4. cleaning_sanitation
  5. general_maintenance

Sanitizes PII, preserves crucial negation and hazard signals ('not', 'no', 'leaking', 'sparking', etc.),
and formats rich records for model training and evaluation.
"""

import re
import os
import argparse
import pandas as pd
import numpy as np

# Deterministic category mapping from multi-attribute complaint types and descriptors
CATEGORY_KEYWORDS = {
    "electrical": [
        "electric", "wiring", "wire", "switch", "mcb", "fuse", "power", "socket",
        "outlet", "spark", "sparking", "short circuit", "voltage", "current",
        "lighting", "light", "bulb", "tube", "fan", "ceiling fan", "appliance", "blackout", "shock"
    ],
    "plumbing": [
        "plumb", "water leak", "leak", "leaking", "pipe", "tap", "faucet", "washbasin",
        "sink", "drain", "drainage", "sewage", "clog", "choked", "toilet", "flush",
        "commode", "geyser", "hot water", "cold water", "overflow", "shower", "jet spray"
    ],
    "furniture_carpentry": [
        "carpentry", "furniture", "door", "window", "lock", "latch", "hinge",
        "handle", "table", "chair", "bed", "cot", "almirah", "cupboard", "drawer",
        "wardrobe", "plywood", "desk", "bolt", "shelf", "wood"
    ],
    "cleaning_sanitation": [
        "sanitation", "unsanitary", "clean", "cleaning", "garbage", "trash", "dustbin",
        "waste", "dirt", "stinking", "odor", "foul smell", "smell", "pest", "cockroach",
        "insect", "rodent", "rat", "mice", "mosquito", "mold", "fungus", "stagnant", "hygiene"
    ],
    "general_maintenance": [
        "paint", "plaster", "peeling", "wall", "crack", "ceiling", "floor", "tiles",
        "seepage", "damp", "elevator", "lift", "railing", "balcony", "stair", "staircase",
        "roof", "masonry", "civil", "glass", "civil work", "corridor"
    ]
}

# Negation and critical safety words that must NEVER be stripped
PRESERVE_TOKENS = {
    "not", "no", "nor", "none", "never", "without", "neither",
    "leak", "leaking", "spark", "sparking", "smoke", "fire",
    "shock", "burst", "urgent", "danger", "hazard", "broken", "overflow"
}


def clean_text_preserve_signals(text: str) -> str:
    """
    Cleans complaint text, removes unnecessary PII (emails, phone numbers, roll numbers),
    normalizes whitespace, but strictly preserves critical negations and urgency signals.
    """
    if not isinstance(text, str):
        return ""
    
    t = text.lower().strip()
    
    # Remove emails
    t = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', t)
    # Remove phone numbers (10 digits or with +91)
    t = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', t)
    # Remove student registration numbers like Reg No: 12412125
    t = re.sub(r'\b(reg|roll|id|emp|std)[\s.:_-]*\d+\b', '', t, flags=re.IGNORECASE)
    # Remove URL links
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    # Remove excessive punctuation while keeping hyphen/apostrophe
    t = re.sub(r'[^\w\s\'-]', ' ', t)
    # Normalize whitespaces
    t = re.sub(r'\s+', ' ', t).strip()
    
    return t


def map_to_target_category(complaint_type: str, descriptor: str, text: str) -> str:
    """
    Deterministically maps complaint attributes to one of the 5 canonical categories.
    """
    combined = f"{str(complaint_type).lower()} {str(descriptor).lower()} {str(text).lower()}"
    
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, kw_list in CATEGORY_KEYWORDS.items():
        for kw in kw_list:
            if re.search(r'\b' + re.escape(kw) + r'\b', combined):
                scores[cat] += 2
            elif kw in combined:
                scores[cat] += 1
                
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] > 0:
        return best_cat
    return "general_maintenance"


def derive_urgency_and_priority(text: str, category: str):
    """
    Calculates initial urgency and risk parameters based on complaint text and category.
    """
    t = text.lower()
    critical_signals = ["spark", "sparking", "shock", "smoke", "fire", "burst", "flooding", "blackout", "gas"]
    high_signals = ["leak", "leaking", "not working", "jammed", "stuck", "choked", "overflow", "severe", "urgent", "dark"]
    
    if any(s in t for s in critical_signals):
        urgency = "Critical"
        urgency_score = 95
    elif any(s in t for s in high_signals):
        urgency = "High"
        urgency_score = 75
    elif "moderate" in t or "intermittent" in t or "loose" in t:
        urgency = "Medium"
        urgency_score = 50
    else:
        urgency = "Low"
        urgency_score = 25
        
    category_risk_map = {
        "electrical": 85,
        "plumbing": 70,
        "general_maintenance": 45,
        "furniture_carpentry": 40,
        "cleaning_sanitation": 35
    }
    c_risk = category_risk_map.get(category, 40)
    
    # Priority formula: P = 0.35*C + 0.30*U + 0.20*R + 0.15*I
    # Let resident-selected urgency match initial urgency cue R = urgency_score
    # Let impact I = 40 (single room base)
    p_score = int(round(0.35 * c_risk + 0.30 * urgency_score + 0.20 * urgency_score + 0.15 * 40))
    p_score = max(5, min(100, p_score))
    
    return urgency, p_score


def preprocess_dataset(input_csv: str = "data/raw/housing_maintenance_complaints.csv",
                       output_csv: str = "data/processed/hostel_complaints_cleaned.csv") -> pd.DataFrame:
    """
    Main preprocessing driver.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    if not os.path.exists(input_csv):
        print(f"File {input_csv} not found. Ingesting baseline or generating mirror data...")
        from src.data.download_dataset import generate_authenticated_mirror
        generate_authenticated_mirror(limit=3000, output_path=input_csv)
        
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} raw records from {input_csv}")
    
    # Build consolidated complaint text
    processed_records = []
    
    for idx, row in df.iterrows():
        comp_type = str(row.get("complaint_type", ""))
        descriptor = str(row.get("descriptor", ""))
        raw_text = str(row.get("resolution_description", ""))
        
        full_text = f"{descriptor}. {raw_text}" if descriptor and descriptor != "nan" else raw_text
        cleaned_text = clean_text_preserve_signals(full_text)
        
        if len(cleaned_text.split()) < 3:
            # Fallback if text is too sparse
            cleaned_text = clean_text_preserve_signals(f"Issue with {comp_type}: {descriptor}")
            
        category = map_to_target_category(comp_type, descriptor, cleaned_text)
        urgency, priority_score = derive_urgency_and_priority(cleaned_text, category)
        
        # Location parsing
        address = str(row.get("incident_address", "Hostel Block A, Room 101"))
        block_match = re.search(r'(Block\s+[A-D]|BH-[1-4]|GH-[1-2])', address, re.IGNORECASE)
        room_match = re.search(r'(Room\s+\d+|\b\d{3}\b)', address, re.IGNORECASE)
        
        block = block_match.group(0) if block_match else f"Block {chr(65 + (idx % 4))}"
        room = room_match.group(0) if room_match else f"Room {101 + (idx % 300)}"
        floor = room[-3] if len(re.findall(r'\d+', room)) > 0 and len(re.findall(r'\d+', room)[0]) >= 3 else "1"
        
        # Team mapping
        team_map = {
            "electrical": "electricians",
            "plumbing": "plumbers",
            "furniture_carpentry": "carpenters",
            "cleaning_sanitation": "housekeeping",
            "general_maintenance": "general_maintenance"
        }
        assigned_team = team_map[category]
        
        # Sentiment heuristic for initialization (real sentiment analyzer will classify later)
        lower_t = cleaned_text.lower()
        if any(w in lower_t for w in ["terrible", "worst", "unacceptable", "furious", "danger", "burst", "sparking", "foul"]):
            sentiment = "negative"
        elif any(w in lower_t for w in ["thank", "fixed", "please check", "good", "appreciated"]):
            sentiment = "positive"
        else:
            sentiment = "neutral"
            
        processed_records.append({
            "complaint_id": f"TCK-{20260000 + idx}",
            "block": block,
            "floor": floor,
            "room": room,
            "complaint_type": comp_type,
            "descriptor": descriptor,
            "description": cleaned_text,
            "category": category,
            "urgency": urgency,
            "priority_score": priority_score,
            "assigned_team": assigned_team,
            "sentiment": sentiment,
            "status": row.get("status", "Closed"),
            "created_at": row.get("created_date", "2026-08-01T10:00:00.000"),
            "resolved_at": row.get("closed_date", "2026-08-01T14:30:00.000")
        })
        
    clean_df = pd.DataFrame(processed_records)
    clean_df.to_csv(output_csv, index=False)
    print(f"Preprocessed dataset saved to {output_csv} ({len(clean_df)} records)")
    print("Class distribution:")
    print(clean_df["category"].value_counts())
    return clean_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess maintenance complaint data.")
    parser.add_argument("--input", type=str, default="data/raw/housing_maintenance_complaints.csv")
    parser.add_argument("--output", type=str, default="data/processed/hostel_complaints_cleaned.csv")
    args = parser.parse_args()
    
    preprocess_dataset(input_csv=args.input, output_csv=args.output)
