"""
Real-World Dataset Ingestion Pipeline
Fetches authentic public housing and building maintenance complaints directly
from NYC Open Data (311 Service Requests & HPD Housing Maintenance).
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.parse
import pandas as pd

API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"


def fetch_real_dataset(limit: int = 10000, raw_output_path: str = "data/raw/housing_maintenance_complaints.csv"):
    """
    Downloads real-world building maintenance complaints from NYC Open Data 311 portal.
    """
    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    all_records = []
    
    batch_size = 5000
    offsets = list(range(0, limit, batch_size))
    
    print(f"Connecting to NYC Open Data 311 API to fetch {limit} real complaints...")
    
    for offset in offsets:
        cur_limit = min(batch_size, limit - len(all_records))
        params = {
            "$select": "unique_key,created_date,closed_date,agency,complaint_type,descriptor,location_type,incident_address,borough,status,resolution_description",
            "$where": "agency in('HPD','DOB') and complaint_type is not null and descriptor is not null",
            "$order": "created_date DESC",
            "$limit": cur_limit,
            "$offset": offset
        }
        url = API_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Project-472-SmartHostel/1.0 (Academic Research Project)"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                all_records.extend(data)
                print(f"Retrieved {len(data)} real complaints (Offset: {offset}, Total: {len(all_records)})")
        except Exception as e:
            print(f"Error fetching batch at offset {offset}: {e}")
            break
            
    if not all_records:
        raise RuntimeError("Failed to fetch real-world records from NYC Open Data.")
        
    df = pd.DataFrame(all_records)
    df.to_csv(raw_output_path, index=False)
    print(f"Successfully saved {len(df)} authentic real-world records to {raw_output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", type=str, default="data/raw/housing_maintenance_complaints.csv")
    args = parser.parse_args()
    
    fetch_real_dataset(limit=args.limit, raw_output_path=args.output)
