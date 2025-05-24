from typing import List, Dict, Any, Optional
from pprint import pprint
from mock_dashboard_api import get_eim_by_id

def get_eim_issues(eim_id: str) -> Optional[Dict[str, Any]]:
    """
    Get issue details for a specific EIM ID directly from mock data
    
    Args:
        eim_id: The EIM ID to query (e.g., "DMOV-1001")
        
    Returns:
        Dictionary containing EIM details including issues, or None if not found
    """
    try:
        # Get the EIM using the direct access function
        eim_data = get_eim_by_id(eim_id)
        if eim_data:
            # Convert Pydantic model to dict for consistent return format
            return eim_data.dict()
        
        # If EIM not found
        print(f"EIM with ID {eim_id} not found")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None

def display_eim_issues(eim_id: str) -> None:
    """
    Display issues for a specific EIM ID in a formatted way
    
    Args:
        eim_id: The EIM ID to query (e.g., "DMOV-1001")
    """
    eim_data = get_eim_issues(eim_id)
    
    if eim_data:
        print(f"Found EIM: {eim_data['eim_id']}")
        print(f"Interface Classification: {eim_data['interface_classification']}")
        print(f"Interface Regular Period: {eim_data['interface_regular_period']}")
        print(f"Dataset ID: {eim_data['dataset_id']}")
        print(f"Dataset Classification: {eim_data['dataset_classification']}")
        print(f"Dataset Regular Period: {eim_data['dataset_regular_period']}")
        
        # Display issue details
        issues = eim_data.get("issues", [])
        if issues:
            print(f"\nFound {len(issues)} issues:")
            for i, issue in enumerate(issues, 1):
                print(f"\nIssue {i}:")
                print(f"  ID: {issue['issue_id']}")
                print(f"  Description: {issue['description']}")
                print(f"  Severity: {issue['severity']}")
                print(f"  Status: {issue['status']}")
                print(f"  Created: {issue['created_date']}")
        else:
            print("\nNo issues found for this EIM")
    else:
        print(f"Could not retrieve data for EIM ID: {eim_id}")

# Example usage
if __name__ == "__main__":
    # Print information about available EIMs for testing
    from mock_dashboard_api import MOCK_DATA
    
    print("Available EIM IDs in the data:")
    for item in MOCK_DATA:
        print(f"- {item.eim_id}")
    print()
    
    # Specify the EIM ID you want to query
    target_eim_id = "DMOV-1004"  # EIM with multiple issues
    print(f"Querying EIM ID: {target_eim_id}")
    
    # Display issues for the specified EIM ID
    display_eim_issues(target_eim_id)
    
    # Test another ID
    print("\n---\n")
    target_eim_id = "DMOV-1003"  # EIM with no issues
    print(f"Querying EIM ID: {target_eim_id}")
    display_eim_issues(target_eim_id)