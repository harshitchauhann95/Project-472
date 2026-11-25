"""
Team Routing and Dispatch Module
Maps AI-predicted complaint categories to maintenance teams:
  - electrical -> electricians
  - plumbing -> plumbers
  - furniture_carpentry -> carpenters
  - cleaning_sanitation -> housekeeping
  - general_maintenance -> general_maintenance

Enforces confidence threshold gating (< 0.60) to prevent incorrect assignments
and routes uncertain complaints to the Warden's 'Needs Review' queue.
"""

from typing import Dict, Any, Optional

TEAM_MAP = {
    "electrical": "electricians",
    "plumbing": "plumbers",
    "furniture_carpentry": "carpenters",
    "cleaning_sanitation": "housekeeping",
    "general_maintenance": "general_maintenance"
}

DISPLAY_TEAM_NAMES = {
    "electricians": "Electrical Maintenance Crew",
    "plumbers": "Plumbing & Sanitary Squad",
    "carpenters": "Carpentry & Furniture Unit",
    "housekeeping": "Housekeeping & Sanitation Team",
    "general_maintenance": "General Facility Maintenance Crew"
}

# Approved confidence threshold from project specification
CONFIDENCE_THRESHOLD = 0.60


class TeamRouter:
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = confidence_threshold

    def route_complaint(self,
                        predicted_category: str,
                        confidence: float,
                        warden_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Determines the assigned team, review gating, and status.
        """
        # 1. Warden manual administrative override
        if warden_override and warden_override.get("enabled"):
            manual_team = warden_override.get("assigned_team")
            if not manual_team and warden_override.get("category"):
                manual_team = TEAM_MAP.get(warden_override["category"], "general_maintenance")
                
            return {
                "assigned_team": manual_team,
                "team_display": DISPLAY_TEAM_NAMES.get(manual_team, manual_team),
                "needs_warden_review": False,
                "routing_reason": "Warden administrative assignment override",
                "confidence_passed": True,
                "confidence": confidence
            }
            
        # 2. Confidence threshold check
        if confidence < self.threshold:
            return {
                "assigned_team": "general_maintenance",
                "team_display": "Pending Warden Review (General Maintenance Hold)",
                "needs_warden_review": True,
                "routing_reason": f"Low prediction confidence ({confidence:.2f} < {self.threshold:.2f}). Flagged to 'Needs Review' queue.",
                "confidence_passed": False,
                "confidence": confidence
            }
            
        # 3. Direct automated routing
        team = TEAM_MAP.get(predicted_category, "general_maintenance")
        return {
            "assigned_team": team,
            "team_display": DISPLAY_TEAM_NAMES.get(team, team),
            "needs_warden_review": False,
            "routing_reason": f"High confidence prediction ({confidence:.2f} >= {self.threshold:.2f}) mapped to {team}.",
            "confidence_passed": True,
            "confidence": confidence
        }
