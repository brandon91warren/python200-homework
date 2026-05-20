import os
import json
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from scipy.stats import pearsonr

from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# --- Lesson 02 ---
# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in degrees Celsius."
                }
            },
            "required": ["celsius"],
            "additionalProperties": False
        }
    }
}


print("--- Lesson 02 Q1 ---")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }
}


def run_agent_time_only(user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools only when they are needed."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[get_current_time_schema],
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = f"Unknown tool: {tool_call.function.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return final_response.choices[0].message.content

    return message.content


# Prediction:
# Calling run_agent_time_only("Convert 100 degrees Celsius to Fahrenheit") should not trigger a tool call
# because the only available tool is get_current_time, and that tool does not help with temperature conversion.
# The model can answer the Celsius conversion itself using basic math.
# I predict only 1 API call will be made because no tool call should be needed.

print("\n--- Lesson 02 Q2 ---")
q2_result = run_agent_time_only("Convert 100 degrees Celsius to Fahrenheit")
print(q2_result)
# My prediction should be correct if the model answers directly without calling get_current_time.


# Q3

def run_agent(user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when they are useful."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    tools = [get_current_time_schema, celsius_to_fahrenheit_schema]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")

            if tool_call.function.name == "get_current_time":
                tool_result = get_current_time()
            elif tool_call.function.name == "celsius_to_fahrenheit":
                tool_result = celsius_to_fahrenheit(**args)
            else:
                tool_result = f"Unknown tool: {tool_call.function.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return final_response.choices[0].message.content

    return message.content


print("\n--- Lesson 02 Q3 ---")
response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# A tool should be called here because celsius_to_fahrenheit is available and directly matches the task.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# A tool may not be called here because the question asks for a plain-English fact, not a direct conversion request.


# --- Lesson 03 ---
# Multi-Tool Agent
# Q4

class CsvManager:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.filename: Optional[str] = None

    def load_csv(self, filename: str):
        try:
            self.df = pd.read_csv(filename)
            self.filename = filename
            return {
                "status": "success",
                "filename": filename,
                "rows": len(self.df),
                "columns": list(self.df.columns)
            }
        except Exception as e:
            return {"error": str(e)}

    def preview_csv(self, rows: int = 5):
        if self.df is None:
            return {"error": "No CSV loaded."}
        return self.df.head(rows).to_dict(orient="records")

    def get_columns(self):
        if self.df is None:
            return {"error": "No CSV loaded."}
        return list(self.df.columns)

    def summarize_csv(self):
        if self.df is None:
            return {"error": "No CSV loaded."}
        return self.df.describe(include="all").to_dict()

    def plot_scatter(self, x_col: str, y_col: str, color: str = "blue"):
        if self.df is None:
            return {"error": "No CSV loaded."}
        if x_col not in self.df.columns or y_col not in self.df.columns:
            return {"error": "One or both columns were not found."}

        import matplotlib.pyplot as plt

        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/scatter_{x_col}_vs_{y_col}.png"

        plt.figure()
        plt.scatter(self.df[x_col], self.df[y_col], color=color)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.savefig(output_path)
        plt.close()

        return {"status": "success", "file": output_path, "color": color}

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return {"error": "No CSV is loaded."}

        if col1 not in self.df.columns:
            return {"error": f"Column not found: {col1}"}

        if col2 not in self.df.columns:
            return {"error": f"Column not found: {col2}"}

        try:
            clean_df = self.df[[col1, col2]].dropna()
            r, p = pearsonr(clean_df[col1], clean_df[col2])
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4)
            }
        except Exception as e:
            return {"error": str(e)}


csv_manager = CsvManager()


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file into memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The path to the CSV file."}
                },
                "required": ["filename"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "preview_csv",
            "description": "Preview the first rows of the loaded CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {"type": "integer", "description": "Number of rows to preview."}
                },
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Return the column names from the loaded CSV file.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_csv",
            "description": "Summarize the loaded CSV file.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_scatter",
            "description": "Create a scatter plot from two columns in the loaded CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_col": {"type": "string", "description": "Column for the x-axis."},
                    "y_col": {"type": "string", "description": "Column for the y-axis."},
                    "color": {"type": "string", "description": "Dot color for the scatter plot."}
                },
                "required": ["x_col", "y_col"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two columns in the loaded CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "The first numeric column."},
                    "col2": {"type": "string", "description": "The second numeric column."}
                },
                "required": ["col1", "col2"],
                "additionalProperties": False
            }
        }
    }
]


node_tools = {
    "load_csv": csv_manager.load_csv,
    "preview_csv": csv_manager.preview_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_csv": csv_manager.summarize_csv,
    "plot_scatter": csv_manager.plot_scatter,
    "compute_correlation": csv_manager.compute_correlation
}


SYSTEM_PROMPT = """
You are a CSV analysis assistant. Use the available tools to inspect, load, summarize,
plot, and analyze CSV data. Follow the ReAct loop: reason about the task, call tools when
needed, inspect tool results, and then give a final answer. Do not claim you computed
something unless you used the tools or have enough information from tool outputs.
"""


def run_agent_cycle(messages, user_prompt: str, max_tool_rounds: int = 8):
    messages.append({"role": "user", "content": user_prompt})

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if tool_name in node_tools:
                tool_result = node_tools[tool_name](**args)
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, default=str)
            })

    return "Tool round limit reached before the agent produced a final answer."


# Q5
print("\n--- Lesson 03 Q5 ---")
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)
print(result)


# Q6
# Role meanings in the ReAct loop:
# system = instructions that control the agent's behavior
# user = the task or question from the user
# assistant = the model's reasoning response and/or tool request
# tool = the result returned by an executed tool call
print("\n--- Lesson 03 Q6 ---")
print(json.dumps(messages, indent=2, default=str))

# --- Lesson 04 ---
# smolagents
# Q7-Q9

try:
    from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

    @tool
    def load_csv(filename: str) -> dict:
        """
        Load a CSV file into memory.

        Args:
            filename: The path to the CSV file.

        Returns:
            A dictionary with the load status, filename, row count, and column names.
        """
        return csv_manager.load_csv(filename)

    @tool
    def preview_csv(rows: int = 5) -> list:
        """
        Preview the first rows of the loaded CSV file.

        Args:
            rows: Number of rows to preview.

        Returns:
            A list of dictionaries representing the first rows of the CSV file.
        """
        return csv_manager.preview_csv(rows)

    @tool
    def get_columns() -> list:
        """
        Return the column names from the loaded CSV file.

        Returns:
            A list of column names from the loaded CSV file.
        """
        return csv_manager.get_columns()

    @tool
    def summarize_csv() -> dict:
        """
        Summarize the loaded CSV file.

        Returns:
            A dictionary containing summary statistics for the loaded CSV file.
        """
        return csv_manager.summarize_csv()

    @tool
    def plot_scatter(x_col: str, y_col: str, color: str = "blue") -> dict:
        """
        Create a scatter plot from the loaded CSV data.

        Args:
            x_col: Column name for the x-axis.
            y_col: Column name for the y-axis.
            color: Color of the scatter plot points.

        Returns:
            A dictionary containing the saved plot filename.
        """
        return csv_manager.plot_scatter(x_col, y_col, color)

    @tool
    def compute_correlation(col1: str, col2: str) -> dict:
        """
        Compute the Pearson correlation coefficient and p-value between two numeric columns.

        Args:
            col1: The first numeric column.
            col2: The second numeric column.

        Returns:
            A dictionary containing the two columns, Pearson r value, and p-value.
        """
        return csv_manager.compute_correlation(col1, col2)

    print("\n--- Lesson 04 Q7 ---")
    print(compute_correlation.description)

    # smolagents automatically generates a tool description from the function name,
    # type hints, docstring, and Args section. In Q4, I manually wrote the JSON schema.
    # To produce a good description, smolagents needs clear function names, type hints,
    # a useful docstring, and clear argument descriptions from the developer.

    print("\n--- Lesson 04 Q8 ---")

    model = OpenAIServerModel(
        model_id=MODEL,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    TOOLS = [
        load_csv,
        preview_csv,
        get_columns,
        summarize_csv,
        plot_scatter,
        compute_correlation
    ]

    tool_agent = ToolCallingAgent(
        tools=TOOLS,
        model=model
    )

    code_agent = CodeAgent(
        tools=TOOLS,
        model=model,
        additional_authorized_imports=[
            "matplotlib",
            "matplotlib.pyplot"
        ]
    )

    prompt = (
        "Load bike_commute.csv. "
        "Plot avg_heart_rate vs duration_min "
        "as a scatter plot with green dots."
    )

    response_tool = tool_agent.run(prompt)

    response_code = code_agent.run(
        prompt,
        additional_args={"csv_manager": csv_manager}
    )

    print("ToolCallingAgent response:", response_tool)
    print("CodeAgent response:", response_code)

    # The ToolCallingAgent successfully used the predefined plot_scatter tool
    # and created a green scatter plot file.
    #
    # The CodeAgent generated and executed Python code dynamically. After
    # allowing matplotlib imports, it was able to create plots using generated code.
    #
    # This reveals that ToolCallingAgent works best when tasks match predefined
    # tools exactly, while CodeAgent is more flexible for open-ended tasks
    # that require custom Python logic.

except Exception as e:
    print("\n--- Lesson 04 Q7-Q8 skipped ---")
    print("smolagents section could not run. Install smolagents and check your OpenAI API key.")
    print("Error:", e)


# Q9

# A ToolCallingAgent would be a better choice than a CodeAgent for tasks like
# loading CSV files, checking columns, computing correlations, or calling APIs
# where the available tools already cover the task requirements.
#
# The key advantage is safety and predictability because the agent can only
# choose from predefined tools with known inputs and outputs.
#
# One meaningful risk of a CodeAgent is that it generates and executes Python code.
# Generated code could contain bugs, overwrite files, run inefficiently, or attempt
# unsafe operations if not properly restricted.
#
# A ToolCallingAgent is more limited and therefore safer because it can only call
# the tools explicitly provided by the developer.