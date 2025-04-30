from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import random
import uvicorn
from datetime import datetime, timedelta
import uuid

# Create application
app = FastAPI(
    title="Mock Dashboard API",
    description="A mock API for dashboard data including EIM information",
    version="1.0.0"
)

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

# Generate random data
def generate_mock_data(count: int = 20):
    mock_data = []
    
    interface_classifications = ["A-Class", "B-Class", "C-Class", "Critical", "Standard"]
    interface_periods = ["Daily", "Weekly", "Monthly", "Quarterly", "On-demand"]
    dataset_classifications = ["Confidential", "Internal", "Public", "Restricted"]
    dataset_periods = ["Real-time", "Hourly", "Daily", "Weekly", "Monthly"]
    issue_severities = ["Critical", "High", "Medium", "Low"]
    issue_statuses = ["Open", "In Progress", "Resolved", "Closed"]
    
    for i in range(count):
        # Generate random date within last 90 days
        random_days = random.randint(0, 90)
        created_date = (datetime.now() - timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        # Generate random issues (0-3 per EIM)
        issues_count = random.randint(0, 3)
        issues = []
        for j in range(issues_count):
            issues.append(Issue(
                issue_id=f"ISS-{uuid.uuid4().hex[:8].upper()}",
                description=f"Issue with data quality in module {random.choice(['A', 'B', 'C', 'D'])}",
                severity=random.choice(issue_severities),
                created_date=created_date,
                status=random.choice(issue_statuses)
            ))
        
        # Generate EIM data
        eim_data = EIMData(
            eim_id=f"EIM-{10000 + i}",
            interface_classification=random.choice(interface_classifications),
            interface_regular_period=random.choice(interface_periods),
            dataset_id=f"DS-{random.randint(1000, 9999)}",
            dataset_classification=random.choice(dataset_classifications),
            dataset_regular_period=random.choice(dataset_periods),
            issues=issues
        )
        mock_data.append(eim_data)
    
    return mock_data

# Initialize mock data
MOCK_DATA = generate_mock_data(30)

# API routes
@app.get("/", tags=["Info"])
async def root():
    return {"message": "Welcome to the Mock Dashboard API"}

@app.get("/api/dashboard/eim", response_model=List[EIMData], tags=["Dashboard"])
async def get_eim_data(
    limit: int = Query(10, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    eim_id: Optional[str] = Query(None, description="Filter by EIM ID"),
    interface_classification: Optional[str] = Query(None, description="Filter by interface classification"),
    dataset_classification: Optional[str] = Query(None, description="Filter by dataset classification"),
    has_issues: Optional[bool] = Query(None, description="Filter to only show EIMs with issues")
):
    """
    Get EIM data with optional filtering parameters
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

@app.get("/api/dashboard/eim/{eim_id}", response_model=EIMData, tags=["Dashboard"])
async def get_eim_by_id(eim_id: str):
    """
    Get detailed information about a specific EIM by ID
    """
    for item in MOCK_DATA:
        if item.eim_id == eim_id:
            return item
    
    raise HTTPException(status_code=404, detail=f"EIM with ID {eim_id} not found")

@app.get("/api/dashboard/issues", response_model=List[Issue], tags=["Dashboard"])
async def get_issues(
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, description="Maximum number of issues to return")
):
    """
    Get a list of issues with optional filtering
    """
    all_issues = []
    for eim in MOCK_DATA:
        all_issues.extend(eim.issues)
    
    # Apply filters
    if severity:
        all_issues = [issue for issue in all_issues if severity.lower() in issue.severity.lower()]
    
    if status:
        all_issues = [issue for issue in all_issues if status.lower() in issue.status.lower()]
    
    # Apply limit
    return all_issues[:limit]

@app.get("/api/dashboard/stats", tags=["Dashboard"])
async def get_dashboard_stats():
    """
    Get summary statistics for the dashboard
    """
    total_eim = len(MOCK_DATA)
    total_issues = sum(len(eim.issues) for eim in MOCK_DATA)
    eim_with_issues = sum(1 for eim in MOCK_DATA if len(eim.issues) > 0)
    
    interface_classifications = {}
    dataset_classifications = {}
    
    for eim in MOCK_DATA:
        if eim.interface_classification in interface_classifications:
            interface_classifications[eim.interface_classification] += 1
        else:
            interface_classifications[eim.interface_classification] = 1
            
        if eim.dataset_classification in dataset_classifications:
            dataset_classifications[eim.dataset_classification] += 1
        else:
            dataset_classifications[eim.dataset_classification] = 1
    
    return {
        "total_eim": total_eim,
        "total_issues": total_issues,
        "eim_with_issues": eim_with_issues,
        "interface_classifications": interface_classifications,
        "dataset_classifications": dataset_classifications
    }

# Start application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 