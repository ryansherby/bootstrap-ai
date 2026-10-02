from pydantic import BaseModel
from typing import List

class FunctionResponse(BaseModel):
    function_name: str
    dependencies: List[str]  # e.g., ["scipy", "scikit-learn"]
    code: str  # The raw string containing the function and docstring
    
    
from google import genai
import os
import pandas as pd

with open("utils/shared_prompt.txt", "r") as f:
    shared_context = f.read()

client = genai.Client(api_key=os.environ.get("GENAI_API_KEY"))

def generate_ready_to_run_function(func_name: str, task_description: str):
    # The prompt combines the shared context with the specific user scenario
    prompt = f"Function Name: {func_name}\nTask: {task_description}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[shared_context, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": FunctionResponse,
            "temperature": 0.0 # High determinism for code generation
        }
    )
    
    return response.parsed


with open("utils/primitives.csv", "r") as f:
    primitives_df = pd.read_csv(f)
    #primitives_df = primitives_df[-1:]
    
    
deps = set(["pandas","numpy"])
funcs = []
    
for index, row in primitives_df.iterrows():
    func_name = row["Tool Name"]
    task_description = f"Tool Definition:{row['Tool Definition']}\n Tool Configuration Parameters:{row['Configuration Parameters']}"
    
    print(f"Generating function for: {func_name}")
    
    function_response = generate_ready_to_run_function(func_name, task_description)

    print(f"Generated function: {function_response.function_name} with dependencies: {function_response.dependencies}")
    
    deps.update(function_response.dependencies)
    funcs.append((function_response.function_name, function_response.code))
    
with open("requirements.txt", "a") as f:
    for dep in deps:
        f.write(f"{dep}\n")
        
for func_name, code in funcs:
    with open(f"library/{func_name}.py", "w") as f:
        f.write(code)
