import sqlite3
import os
import json
from openai import OpenAI
from bfs import solve

conn = sqlite3.connect('movielens.db')

def get_table_names():
    table_names = []
    tables = conn.execute('Select name from sqlite_master where type="table"')
    for table in tables.fetchall():
        table_names.append(table[0])

    return table_names

def get_column_names(table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names


def get_database_info():
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names():
        columns_names = get_column_names(table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts

database_schema = get_database_info()

database_schema_string = "\n".join(
    f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}" for table in database_schema
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "Use this function to answer user questions about music. Input should be a fully formed SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
                                {database_schema_string}
                                The query should be returned in plain text, not in JSON.
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "solve_task",
            "description": "Use this function to solve a task using the BFS algorithm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "object",
                        "description": "The task object to be solved."
                    },
                    "idx": {
                        "type": "integer",
                        "description": "The index of the input to be solved."
                    }
                },
                "required": ["task", "idx"]
            }
        }
    }
]

def ask_database(query):
    results = str(conn.execute(query).fetchall())
    return results

def solve_task(task, idx):
    return solve(args, task, idx, to_print=True)

client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))

# Create the chat completion
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Answer user questions by generating SQL queries, when selecting title use 'True' or 'False' (with quotes) instead of 1 or 0 "
        },
        {
            "role": "user",
            "content": "Recommend 3 different comedies with highest rating"
        }
    ],
    model="gpt-3.5-turbo",
    tools=tools
)

# Get the tool call from the chat completion
tool_call = chat_completion.choices[0].message.tool_calls[0]

# Check if the tool call is for the database or the solve_task function
if tool_call.function.name == 'ask_database':
    # Extract the query from the tool call arguments
    query = json.loads(tool_call.function.arguments)["query"]
    # Execute the query and print the result
    result = ask_database(query)
    print(result)
elif tool_call.function.name == 'solve_task':
    # Extract the task and index from the tool call arguments
    task = json.loads(tool_call.function.arguments)["task"]
    idx = json.loads(tool_call.function.arguments)["idx"]
    # Call the solve_task function and print the result
    result = solve_task(task, idx)
    print(result)


def main():
    # logs
    print("This is the main function in new.py")
    print(logging.info("Feature log: Data preprocessing completed."))
    print(logging.info("Feature log: Model training started."))
    print(logging.info("Feature log: Hyperparameter tuning in progress."))
    print(logging.info("Feature log: Model evaluation and results analysis."))
    print(logging.info("Feature log: Experiment completed successfully."))

if __name__ == "__main__":
    main()





