"""
Priority Scoring Engine
Computes transparent priority score based on Formula (1) & (2) from Smart_Hostel_Report_File.pdf:
  P = 0.35*C + 0.30*U + 0.20*R + 0.15*I  (0–100 scale)
  P_band = High (P >= 75), Medium (40 <= P < 75), Low (P < 40)

Features:
- Category Risk (C)
- Urgency Cue Strength (U) from text signals
- Resident-Selected Urgency (R)
- Impact Scope (I)
- Automatic Hazard Escalation Override (active danger/fire/sparking)
- Warden Override Support
"""

import re
from typing import Dict, Any, Optional

# Default category risk weights (0-100 scale)
CATEGORY_RISK_MAP = {
    "electrical": 85,
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

# Immediate safety hazard keywords that trigger rule-based escalation
HAZARD_KEYWORDS = [
    "spark", "sparking", "smoke", "fire", "live wire", "burst pipe",
    "electric shock", "gas smell", "burning", "ceiling collapse", "flooding room"
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
        
        # Hazard cues (Score 95)
        for hk in HAZARD_KEYWORDS:
            if hk in lower_t:
                detected_cues.append(hk)
                
        if detected_cues:
            return 95, detected_cues
            
        # High urgency cues (Score 75)
        high_cues = ["leak", "leaking", "overflow", "not working", "jammed", "choked", "dark", "urgent", "broken"]
        for hc in high_cues:
            if re.search(r'\b' + re.escape(hc) + r'\b', lower_t):
                detected_cues.append(hc)
                
        if detected_cues:
            return 75, detected_cues
            
        # Medium urgency cues (Score 50)
        med_cues = ["loose", "intermittent", "noisy", "rattling", "flicker", "slow", "blinking"]
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
        
        # Formula: P = 0.35*C + 0.30*U + 0.20*R + 0.15*I
        raw_score = (self.w_c * c_score) + (self.w_u * u_score) + (self.w_r * r_score) + (self.w_i * i_score)
        p_score = int(round(raw_score))
        p_score = max(5, min(100, p_score))
        
        # Safety hazard rule escalation
        hazard_triggered = False
        hazard_reason = None
        lower_desc = description.lower()
        for hk in HAZARD_KEYWORDS:
            if hk in lower_desc:
                hazard_triggered = True
                hazard_reason = f"Safety rule escalation: detected critical hazard cue '{hk}'"
                if p_score < 80:
                    p_score = 85
                break
                
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


# Module self-test
if __name__ == "__main__":
    engine = PriorityEngine()
    test_cases = [
        ("electrical", "Ceiling fan sparking violently and smelling like burnt plastic", "Emergency", "Shared Room (2-4 Residents)"),
        ("plumbing", "Tap dripping small drops in corner washbasin", "Low", "Single Resident Room"),
        ("furniture_carpentry", "Study chair backrest cracked", "Medium", "Single Resident Room")
    ]
    for cat, text, urg, imp in test_cases:
        res = engine.compute_priority(cat, text, urg, imp)
        print(f"[{cat.upper()}] P={res['priority_score']} ({res['priority_band']}) | Escalated: {res['hazard_escalation']}")
