import os
import ast
import operator

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent


# ============================================================
# 1. LOAD GROQ API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please add your Groq API key in the .env file."
    )


# ============================================================
# 2. STUDENT DATABASE
# ============================================================

students = {
    "101": {
        "name": "Ashmi",
        "department": "CSE",
        "attendance": 82
    },

    "102": {
        "name": "Akshith",
        "department": "CSE",
        "attendance": 91
    },

    "103": {
        "name": "Steffi",
        "department": "CSE",
        "attendance": 88
    },

    "104": {
        "name": "Ashmica",
        "department": "CSE",
        "attendance": 76
    }
}


# ============================================================
# 3. CALCULATOR HELPER
# ============================================================

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow
}


def calculate_expression(node):

    if isinstance(node, ast.Expression):
        return calculate_expression(node.body)

    if isinstance(node, ast.Constant):

        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):

        left = calculate_expression(node.left)
        right = calculate_expression(node.right)

        operation = allowed_operators.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported mathematical operation.")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):

        value = calculate_expression(node.operand)

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return value

    raise ValueError("Invalid expression.")


# ============================================================
# 4. TOOL 1 — CALCULATOR
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate basic mathematical expressions.

    Supports:
    Addition
    Subtraction
    Multiplication
    Division
    Percentage calculations
    """

    try:

        expression = expression.replace("×", "*")
        expression = expression.replace("÷", "/")

        tree = ast.parse(expression, mode="eval")

        result = calculate_expression(tree)

        if isinstance(result, float):
            result = round(result, 2)

        return str(result)

    except ZeroDivisionError:

        return "Cannot divide by zero."

    except Exception:

        return "Invalid mathematical expression."


# ============================================================
# 5. TOOL 2 — STUDENT INFORMATION
# ============================================================

@tool
def student_information(name: str) -> str:
    """
    Find a student's department and basic information.
    """

    name = name.strip().lower()

    for roll_number, student in students.items():

        if student["name"].lower() == name:

            return (
                f"Name: {student['name']}\n"
                f"Roll Number: {roll_number}\n"
                f"Department: {student['department']}"
            )

    return (
        f"I couldn't find {name.title()} "
        "in the student database."
    )


# ============================================================
# 6. TOOL 3 — ATTENDANCE
# ============================================================

@tool
def attendance(name: str) -> str:
    """
    Retrieve a student's attendance percentage.
    """

    name = name.strip().lower()

    for student in students.values():

        if student["name"].lower() == name:

            return (
                f"{student['name']}'s attendance is "
                f"{student['attendance']}%."
            )

    return (
        f"I couldn't find {name.title()} "
        "in the student database."
    )


# ============================================================
# 7. GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# 8. TOOLS GIVEN TO THE AGENT
# ============================================================

tools = [
    calculator,
    student_information,
    attendance
]


# ============================================================
# 9. AGENT INSTRUCTIONS
# ============================================================

system_prompt = """
You are an Agentic AI Student Assistant.

Your main job is to understand the user's request
and decide which tool should be used.

IMPORTANT:
Do NOT manually hard-code the conversation flow.
The agent must decide which tool is appropriate
based on the user's request.

AVAILABLE TOOLS:

1. calculator

Use the calculator tool for mathematical questions.

Examples:
- Calculate 25 * 40
- Calculate 250 * 15 / 100
- What is 50 + 20?


2. student_information

Use this tool when the user asks for information
about a student.

The student database contains:

Ashmi:
Roll Number: 101
Department: CSE

Akshith:
Roll Number: 102
Department: CSE

Steffi:
Roll Number: 103
Department: CSE

Ashmica:
Roll Number: 104
Department: CSE


3. attendance

Use this tool when the user asks about a student's
attendance.

Current attendance:

Ashmi: 82%
Akshith: 91%
Steffi: 88%
Ashmica: 76%


UNKNOWN STUDENT:

If the requested student is not present in the
database, respond politely:

"I couldn't find [name] in the student database."

The application must not crash.


GENERAL QUESTIONS:

If the user asks a general question such as:

"What is Python?"

Answer using the LLM.

Do not unnecessarily call the student tools.


IMPORTANT:

Always decide the appropriate tool based on
the user's request.

Give simple and clear answers.
"""


# ============================================================
# 10. CREATE THE AI AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# ============================================================
# 11. FUNCTION TO ASK THE AGENT
# ============================================================

def ask_agent(user_question):

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_question
                }
            ]
        }
    )

    return response["messages"][-1].content


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

print("=" * 60)
print("       AGENTIC AI - STUDENT ASSISTANT")
print("=" * 60)

print("""
Available features:

1. Calculator
2. Student Information
3. Attendance
4. General AI Questions

Students in database:

101 - Ashmi
102 - Akshith
103 - Steffi
104 - Ashmica

Type 'exit' to stop.
""")


# ============================================================
# 13. CHAT LOOP
# ============================================================

while True:

    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:

        print("\nAssistant: Goodbye! 👋")
        break

    if user_input == "":
        continue

    try:

        answer = ask_agent(user_input)

        print("\nAssistant:", answer)

    except Exception as error:

        print("\nAssistant: Sorry, something went wrong.")

        print("Error:", error)