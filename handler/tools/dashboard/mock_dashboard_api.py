from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
from datetime import datetime

# Define data models
class Issue(BaseModel):
    issue_id: str = Field(..., description="Unique identifier for the issue")
    description: str = Field(..., description="Description of the issue")
    severity: str = Field(..., description="Severity level of the issue")
    created_date: str = Field(..., description="Date when the issue was created")
    status: str = Field(..., description="Current status of the issue")

class EIMData(BaseModel):
    eim_id: str = Field(..., description="EIM/Application ID")
    interface_classification: str = Field(..., description="Classification of the interface")
    interface_regular_period: str = Field(..., description="Regular period for the interface")
    dataset_id: str = Field(..., description="Dataset identifier")
    dataset_classification: str = Field(..., description="Classification of the dataset")
    dataset_regular_period: str = Field(..., description="Regular period for the dataset")
    issues: List[Issue] = Field(default_factory=list, description="List of issues associated with this EIM")

# Load data from JSON file
def load_data_from_json():
    try:
        # Get the project root directory (3 levels up from current file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        # Construct path to JSON file in project root's data directory
        json_file_path = os.path.join(project_root, "data", "dmov_1.json")
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found at {json_file_path}")
            
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Convert JSON data to EIMData objects
        mock_data = []
        for item in data:
            # Convert issues to Issue objects
            issues = []
            for issue_data in item.get('issues', []):
                issues.append(Issue(**issue_data))
                
            # Create EIMData object
            eim_data = EIMData(
                eim_id=item['eim_id'],
                interface_classification=item['interface_classification'],
                interface_regular_period=item['interface_regular_period'],
                dataset_id=item['dataset_id'],
                dataset_classification=item['dataset_classification'],
                dataset_regular_period=item['dataset_regular_period'],
                issues=issues
            )
            mock_data.append(eim_data)
            
        print(f"Loaded {len(mock_data)} EIM records from {json_file_path}")
        return mock_data
    except Exception as e:
        print(f"Error loading data from JSON: {e}")
        return []

# Initialize mock data from JSON file
MOCK_DATA = load_data_from_json()

# Direct access functions
def get_eim_data(
    limit: int = 10, 
    offset: int = 0,
    eim_id: Optional[str] = None,
    interface_classification: Optional[str] = None,
    dataset_classification: Optional[str] = None,
    has_issues: Optional[bool] = None
) -> List[EIMData]:
    """
    Get EIM data with optional filtering parameters
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        eim_id: Filter by EIM ID
        interface_classification: Filter by interface classification
        dataset_classification: Filter by dataset classification
        has_issues: Filter to only show EIMs with issues
        
    Returns:
        List of filtered EIMData objects
    """
    # Apply filters
    filtered_data = MOCK_DATA
    
    if eim_id:
        filtered_data = [item for item in filtered_data if eim_id.lower() in item.eim_id.lower()]
    
    if interface_classification:
        filtered_data = [item for item in filtered_data if interface_classification.lower() in item.interface_classification.lower()]
    
    if dataset_classification:
        filtered_data = [item for item in filtered_data if dataset_classification.lower() in item.dataset_classification.lower()]
    
    if has_issues is not None:
        if has_issues:
            filtered_data = [item for item in filtered_data if len(item.issues) > 0]
        else:
            filtered_data = [item for item in filtered_data if len(item.issues) == 0]
    
    # Apply pagination
    paginated_data = filtered_data[offset:offset+limit]
    
    return paginated_data

def get_eim_by_id(eim_id: str) -> Optional[EIMData]:
    """
    Get detailed information about a specific EIM by ID
    
    Args:
        eim_id: The EIM ID to find
        
    Returns:
        EIMData object if found, None otherwise
    """
    for item in MOCK_DATA:
        if item.eim_id == eim_id:
            return item
    
    return None


# Simple test code
if __name__ == "__main__":
    # Simple test of basic functionality
    print("Starting mock data test...")
    print(f"Loaded {len(MOCK_DATA)} EIM records")
    
    if MOCK_DATA:
        print(f"Sample record: {MOCK_DATA[0].dict()}")
        
        # Test filtering
        print("\nTesting data filtering:")
        filtered = get_eim_data(limit=2, has_issues=True)
        print(f"Found {len(filtered)} EIMs with issues (limited to 2)")
        
        # Test finding by ID
        test_id = MOCK_DATA[0].eim_id
        print(f"\nTesting get_eim_by_id with ID: {test_id}")
        found_eim = get_eim_by_id(test_id)
        if found_eim:
            print(f"Found EIM with {len(found_eim.issues)} issues")
        else:
            print("EIM not found")