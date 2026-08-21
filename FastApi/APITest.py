"""Requirements
Create a Pydantic model called GreetingRequest.
It should have one field:
name: str
Create a POST endpoint /greet.
Inside the endpoint, use the value from the request to build the greeting.
Return the greeting as JSON.
Bonus Challenge ⭐

Modify the request model to include an age.

The client sends:

{
    "name": "Pelumi",
    "age": 18
}

Return:

{
    "message": "Hello Pelumi!",
    "age": 18,
    "status": "Adult"
}

If the age is under 18, return "Minor" instead.

Extra Bonus ⭐⭐

Instead of hardcoding "Welcome to FastAPI.", create a helper function:

def create_greeting(name: str) -> str:
    ...

Call that function from your endpoint. This mirrors how your /ask endpoint calls prompt() and search().

This exercise will reinforce:

✅ Creating a BaseModel
✅ Using POST
✅ Accessing request data with request.name
✅ Returning JSON
✅ Separating logic into helper functions

Give it a try without looking anything up. Paste your code here when you're done, and I'll review it like I would in a code review—pointing out what's good and suggesting improvements where needed."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

 
class GreetingRequest(BaseModel):
    name : str
    age : int

@app.post("/greet")
def build_greeting(request : GreetingRequest):
    if request.age >= 18 :
       Status = "Adult"
    else:
       Status = "Minor"
       return{ "name" : f"Hello {request.name}",
            "age" : request.age,
            "Status" : Status
       }
