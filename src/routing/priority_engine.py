"""
Priority Scoring Engine
Computes transparent priority score based on multi-parameter utility formula:
  P = 0.35*C + 0.30*U + 0.20*R + 0.15*I  (0–100 scale)
  P_band = High (P >= 75), Medium (40 <= P < 75), Low (P < 40)

Features:
- Category Risk (C)
- Urgency Cue Strength (U) from text signals
- Resident-Selected Urgency (R)
- Impact Scope (I)
- Comprehensive Safety Hazard Override: Catches shock, electrocution, sparking, fire, gas, bursting
- Human-in-the-Loop Warden Override
"""

import re
from typing import Dict, Any, Optional

# Default category risk weights (0-100 scale)
CATEGORY_RISK_MAP = {
    "electrical": 90,
    "plumbing": 70,
    "cleaning_sanitation": 45,
    "furniture_carpentry": 40,
    "general_maintenance": 40
}

# Resident urgency to 0-100 score
RESIDENT_URGENCY_MAP = {
    "Emergency": 100,
    "Critical": 95,
    "High": 75,
    "Medium": 50,
    "Low": 25
}

# Impact scope to 0-100 score
IMPACT_SCOPE_MAP = {
    "Hostel / Entire Wing": 100,
    "Common Area / Floor Washroom / Mess": 75,
    "Shared Room (2-4 Residents)": 50,
    "Single Resident Room": 25
}

# Critical Life-Safety & Hazard Keywords
# Any mention of these terms MUST automatically trigger High Priority (>=90)
HAZARD_KEYWORDS = [
    # Electrical hazard & shock signals
    "shock", "shocked", "shocks", "shocking", "electric shock", "electrocuted", "electrocution",
    "earthing", "earthing fault", "current", "current coming", "got shock", "got shocked",
    "spark", "sparking", "sparks", "smoke", "fire", "flame", "burning", "burnt smell",
    "short circuit", "live wire", "exposed wire", "open wire", "wire sparking",
    "high voltage", "power surge", "mcb explosion", "socket blast",
    # Plumbing hazard signals
    "burst pipe", "pipe burst", "flooding", "water flooding", "sewage overflow",
    # Structural & Gas hazard signals
    "ceiling collapse", "balcony broken", "gas leak", "gas smell", "cylinder leak"
]


class PriorityEngine:
    def __init__(self,
                 w_category: float = 0.35,
                 w_urgency_cue: float = 0.30,
                 w_resident: float = 0.20,
                 w_impact: float = 0.15):
        self.w_c = w_category
        self.w_u = w_urgency_cue
        self.w_r = w_resident
        self.w_i = w_impact

    def extract_urgency_cue_score(self, text: str) -> tuple[int, list[str]]:
        """
        Extracts urgency cue strength (0-100) and detected keywords from complaint description.
        """
        if not text:
            return 25, []
            
        lower_t = text.lower()
        detected_cues = []
        
        # 1. Critical Safety Hazards (Score 95-100)
        for hk in HAZARD_KEYWORDS:
            if re.search(r'\b' + re.escape(hk) + r'\b', lower_t) or hk in lower_t:
                detected_cues.append(hk)
                
        if detected_cues:
            return 95, detected_cues
            
        # 2. High Urgency Cues (Score 75)
        high_cues = [
            "leak", "leaking", "overflow", "not working", "jammed", "choked", "clogged",
            "dark", "pitch dark", "urgent", "immediately", "broken", "severe", "tripping"
        ]
        for hc in high_cues:
            if re.search(r'\b' + re.escape(hc) + r'\b', lower_t):
                detected_cues.append(hc)
                
        if detected_cues:
            return 75, detected_cues
            
        # 3. Medium Urgency Cues (Score 50)
        med_cues = ["loose", "intermittent", "noisy", "rattling", "flicker", "slow", "blinking", "creaking"]
        for mc in med_cues:
            if re.search(r'\b' + re.escape(mc) + r'\b', lower_t):
                detected_cues.append(mc)
                
        if detected_cues:
            return 50, detected_cues
            
        return 25, ["routine_report"]

    def compute_priority(self,
                         category: str,
                         description: str,
                         resident_urgency: str = "Medium",
                         impact_scope: str = "Shared Room (2-4 Residents)",
                         warden_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Computes the multi-parameter priority score, band, and safety escalations.
        """
        # Warden manual override check
        if warden_override and warden_override.get("enabled"):
            manual_score = int(warden_override.get("priority_score", 75))
            manual_band = warden_override.get("priority_band", "High")
            return {
                "priority_score": manual_score,
                "priority_band": manual_band,
                "components": {
                    "category_risk_C": None,
                    "urgency_cue_U": None,
                    "resident_urgency_R": None,
                    "impact_I": None
                },
                "hazard_escalation": False,
                "override_applied": True,
                "override_reason": warden_override.get("reason", "Warden manual administrative decision")
            }
            
        # 1. Category Risk (C)
        c_score = CATEGORY_RISK_MAP.get(category, 40)
        
        # 2. Urgency Cue (U)
        u_score, cues = self.extract_urgency_cue_score(description)
        
        # 3. Resident Selected Urgency (R)
        r_score = RESIDENT_URGENCY_MAP.get(resident_urgency, 50)
        
        # 4. Impact Scope (I)
        i_score = IMPACT_SCOPE_MAP.get(impact_scope, 50)
        
        # Safety hazard rule escalation check
        hazard_triggered = False
        hazard_reason = None
        lower_desc = description.lower()
        
        for hk in HAZARD_KEYWORDS:
            if re.search(r'\b' + re.escape(hk) + r'\b', lower_desc) or hk in lower_desc:
                hazard_triggered = True
                hazard_reason = f"CRITICAL SAFETY ESCALATION: Detected active hazard '{hk}'"
                # If hazard is active, category risk for life safety is bumped
                if "shock" in hk or "electrocuted" in hk or "earthing" in hk or "spark" in hk or "wire" in hk or "current" in hk:
                    c_score = max(c_score, 90)
                u_score = 95
                break
                
        # Formula: P = 0.35*C + 0.30*U + 0.20*R + 0.15*I
        raw_score = (self.w_c * c_score) + (self.w_u * u_score) + (self.w_r * r_score) + (self.w_i * i_score)
        p_score = int(round(raw_score))
        
        # If active hazard is detected, priority MUST NEVER drop below 90
        if hazard_triggered:
            p_score = max(p_score, 90)
            
        p_score = max(5, min(100, p_score))
        
        # Band determination: High (>=75), Medium (40-74), Low (<40)
        if p_score >= 75:
            band = "High"
        elif p_score >= 40:
            band = "Medium"
        else:
            band = "Low"
            
        return {
            "priority_score": p_score,
            "priority_band": band,
            "components": {
                "category_risk_C": c_score,
                "urgency_cue_U": u_score,
                "resident_urgency_R": r_score,
                "impact_I": i_score
            },
            "urgency_cues": cues,
            "hazard_escalation": hazard_triggered,
            "hazard_reason": hazard_reason,
            "override_applied": False
        }
