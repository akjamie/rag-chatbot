import requests
from typing import List, Dict, Any, Optional
from pprint import pprint

def get_eim_issues(eim_id: str, api_base_url: str = "http://localhost:8000") -> Optional[Dict[str, Any]]:
    """
    Query the Dashboard API to get issue details for a specific EIM ID
    
    Args:
        eim_id: The EIM ID to query (e.g., "EIM-10001")
        api_base_url: Base URL of the dashboard API
        
    Returns:
        Dictionary containing EIM details including issues, or None if not found
    """
    # Build the API endpoint URL for specific EIM
    url = f"{api_base_url}/api/dashboard/eim/{eim_id}"
    
    try:
        # Make GET request to the API
        response = requests.get(url)
        
        # Raise an exception for 4XX/5XX responses
        response.raise_for_status()
        
        # Parse the JSON response
        eim_data = response.json()
        
        # Return the full EIM data which includes the issues
        return eim_data
        
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            print(f"EIM with ID {eim_id} not found")
        else:
            print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect to API at {api_base_url}")
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None

def display_eim_issues(eim_id: str, api_base_url: str = "http://localhost:8000") -> None:
    """
    Display issues for a specific EIM ID in a formatted way
    
    Args:
        eim_id: The EIM ID to query (e.g., "EIM-10001")
        api_base_url: Base URL of the dashboard API
    """
    eim_data = get_eim_issues(eim_id, api_base_url)
    
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
        print(f"Could not retrieve data for EIM ID: {target_eim_id}")

# Example usage
if __name__ == "__main__":
    # Specify the EIM ID you want to query
    target_eim_id = "EIM-10001"
    
    # Display issues for the specified EIM ID
    display_eim_issues(target_eim_id)