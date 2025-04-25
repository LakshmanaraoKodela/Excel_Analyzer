import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import csv
import io

# Load environment variables
load_dotenv()

# Set the page title and layout
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

# Initialize Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Define the prompt template for generating realistic data
prompt_template = """
Generate realistic synthetic data for the following column:
Column Name: {column_name}
Data Type: {data_type}
Additional Context: {context}
{rules}

Generate {num_rows} unique values that are realistic for this column.
Return only the data in a simple list format, one value per line.
"""

prompt = PromptTemplate(
    input_variables=["column_name", "data_type", "context", "rules", "num_rows"],
    template=prompt_template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Main app title
st.title("Synthetic Data Generator")
st.write("Generate realistic synthetic data using AI")


# Function to generate synthetic data based on column types
def generate_data(column_specs, num_rows):
    data = {}

    # First pass: Generate data for columns without dependencies
    for col_spec in column_specs:
        col_name = col_spec["name"]
        col_type = col_spec["type"]

        # Skip columns with dependencies in the first pass
        if col_spec.get("has_rule", False) and col_spec.get("rule", "").strip():
            continue

        generate_column_data(col_spec, num_rows, data)

    # Second pass: Generate data for columns with dependencies
    for col_spec in column_specs:
        if col_spec.get("has_rule", False) and col_spec.get("rule", "").strip():
            col_name = col_spec["name"]

            # Format rule with actual data from other columns
            rule = col_spec["rule"]

            # Include rule in the context for AI
            col_spec["rule_context"] = f"Additional rule: {rule}"

            generate_column_data(col_spec, num_rows, data)

    return pd.DataFrame(data)


def generate_column_data(col_spec, num_rows, data):
    col_name = col_spec["name"]
    col_type = col_spec["type"]

    # Prepare rules text if any
    rules_text = ""
    if col_spec.get("has_rule", False) and col_spec.get("rule", "").strip():
        rules_text = f"Rules: {col_spec['rule']}"

    if col_type == "Integer":
        min_val = col_spec.get("min", 0)
        max_val = col_spec.get("max", 100)

        # For integers, we'll use AI to generate realistic values within the range
        context = f"Generate integer values between {min_val} and {max_val} that would be realistic for a column named {col_name}."

        # Add dependency context if this column has a rule
        if col_spec.get("rule_context"):
            context += " " + col_spec.get("rule_context")

        try:
            result = chain.run(column_name=col_name, data_type="Integer", context=context, rules=rules_text,
                               num_rows=num_rows)
            values = [int(x.strip()) for x in result.strip().split('\n') if x.strip()]
            # Ensure values are within range and we have enough
            values = [max(min_val, min(max_val, v)) for v in values]
            while len(values) < num_rows:
                values.append(random.randint(min_val, max_val))
            data[col_name] = values[:num_rows]
        except Exception as e:
            st.error(f"Error generating AI data for {col_name}: {e}")
            # Fallback to random generation
            data[col_name] = [random.randint(min_val, max_val) for _ in range(num_rows)]

    elif col_type == "Float":
        min_val = col_spec.get("min", 0.0)
        max_val = col_spec.get("max", 100.0)
        decimals = col_spec.get("decimals", 2)

        context = f"Generate float values between {min_val} and {max_val} with {decimals} decimal places that would be realistic for a column named {col_name}."

        # Add dependency context if this column has a rule
        if col_spec.get("rule_context"):
            context += " " + col_spec.get("rule_context")

        try:
            result = chain.run(column_name=col_name, data_type="Float", context=context, rules=rules_text,
                               num_rows=num_rows)
            values = []
            for x in result.strip().split('\n'):
                try:
                    if x.strip():
                        values.append(round(float(x.strip()), decimals))
                except ValueError:
                    continue
            # Ensure values are within range and we have enough
            values = [max(min_val, min(max_val, v)) for v in values]
            while len(values) < num_rows:
                values.append(round(random.uniform(min_val, max_val), decimals))
            data[col_name] = values[:num_rows]
        except Exception as e:
            st.error(f"Error generating AI data for {col_name}: {e}")
            # Fallback to random generation
            data[col_name] = [round(random.uniform(min_val, max_val), decimals) for _ in range(num_rows)]

    elif col_type == "String":
        context = f"Generate realistic string values for a column named {col_name}."

        # Add dependency context if this column has a rule
        if col_spec.get("rule_context"):
            context += " " + col_spec.get("rule_context")

        try:
            result = chain.run(column_name=col_name, data_type="String", context=context, rules=rules_text,
                               num_rows=num_rows)
            values = [x.strip() for x in result.strip().split('\n') if x.strip()]
            while len(values) < num_rows:
                values.append(f"{col_name}_{len(values)}")
            data[col_name] = values[:num_rows]
        except Exception as e:
            st.error(f"Error generating AI data for {col_name}: {e}")
            # Fallback to random generation
            data[col_name] = [f"{col_name}_{i}" for i in range(num_rows)]

    elif col_type == "Date":
        start_date = col_spec.get("start_date", datetime.date.today() - datetime.timedelta(days=365))
        end_date = col_spec.get("end_date", datetime.date.today())

        # Convert string dates to datetime objects if needed
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        delta = (end_date - start_date).days

        context = f"Generate realistic dates between {start_date} and {end_date} for a column named {col_name}."

        # Add dependency context if this column has a rule
        if col_spec.get("rule_context"):
            context += " " + col_spec.get("rule_context")

        try:
            result = chain.run(column_name=col_name, data_type="Date", context=context, rules=rules_text,
                               num_rows=num_rows)
            values = []
            for x in result.strip().split('\n'):
                try:
                    if x.strip():
                        date_val = datetime.datetime.strptime(x.strip(), "%Y-%m-%d").date()
                        values.append(date_val)
                except ValueError:
                    continue
            # Ensure we have enough dates
            while len(values) < num_rows:
                random_days = random.randint(0, delta)
                values.append(start_date + datetime.timedelta(days=random_days))
            data[col_name] = values[:num_rows]
        except Exception as e:
            st.error(f"Error generating AI data for {col_name}: {e}")
            # Fallback to random generation
            data[col_name] = [start_date + datetime.timedelta(days=random.randint(0, delta)) for _ in
                              range(num_rows)]

    elif col_type == "Boolean":
        context = f"Generate realistic boolean values (True/False) for a column named {col_name}."

        # Add dependency context if this column has a rule
        if col_spec.get("rule_context"):
            context += " " + col_spec.get("rule_context")

        try:
            result = chain.run(column_name=col_name, data_type="Boolean", context=context, rules=rules_text,
                               num_rows=num_rows)
            values = []
            for x in result.strip().split('\n'):
                if x.strip().lower() in ('true', 'false', '1', '0', 'yes', 'no'):
                    values.append(x.strip().lower() in ('true', '1', 'yes'))
            # Ensure we have enough values
            while len(values) < num_rows:
                values.append(random.choice([True, False]))
            data[col_name] = values[:num_rows]
        except Exception as e:
            st.error(f"Error generating AI data for {col_name}: {e}")
            # Fallback to random generation
            data[col_name] = [random.choice([True, False]) for _ in range(num_rows)]

    elif col_type == "Categorical":
        categories = col_spec.get("categories", [])

        if categories:
            context = f"Generate realistic categorical values from the following categories: {', '.join(categories)} for a column named {col_name}."
        else:
            context = f"Generate realistic categorical values for a column named {col_name}."

        # Add dependency context if this column has a rule
        if col_spec.get("rule_context"):
            context += " " + col_spec.get("rule_context")

        try:
            result = chain.run(column_name=col_name, data_type="Categorical", context=context, rules=rules_text,
                               num_rows=num_rows)
            values = [x.strip() for x in result.strip().split('\n') if x.strip()]

            # If categories were provided, ensure values are from those categories
            if categories:
                values = [v if v in categories else random.choice(categories) for v in values]

            # Ensure we have enough values
            while len(values) < num_rows:
                if categories:
                    values.append(random.choice(categories))
                else:
                    values.append(f"Category_{len(values)}")
            data[col_name] = values[:num_rows]
        except Exception as e:
            st.error(f"Error generating AI data for {col_name}: {e}")
            # Fallback to random generation
            if categories:
                data[col_name] = [random.choice(categories) for _ in range(num_rows)]
            else:
                data[col_name] = [f"Category_{i % 5}" for i in range(num_rows)]


# Function to convert dataframe to CSV
def convert_df_to_csv(df):
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


# Main application layout
num_rows = st.number_input("Number of rows to generate", min_value=1, max_value=10000000, value=100)

# Initialize session state for columns if it doesn't exist
if 'columns' not in st.session_state:
    st.session_state.columns = [{"name": "Column1", "type": "String", "has_rule": False, "rule": ""}]

# Container for columns
columns_container = st.container()


# Function to add a new column
def add_column():
    st.session_state.columns.append(
        {"name": f"Column{len(st.session_state.columns) + 1}", "type": "String", "has_rule": False, "rule": ""})


# Function to remove a column
def remove_column(index):
    st.session_state.columns.pop(index)


# Function to toggle rule for a column
def toggle_rule(index):
    st.session_state.columns[index]["has_rule"] = not st.session_state.columns[index]["has_rule"]


# Display column configuration
with columns_container:
    st.subheader("Column Configuration")

    # Loop through each column in session state
    for i, col in enumerate(st.session_state.columns):
        col1, col2, col3, col4, col5 = st.columns([2, 2, 5, 1, 1])

        with col1:
            st.session_state.columns[i]["name"] = st.text_input(f"Column Name", value=col["name"], key=f"name_{i}")

        with col2:
            col_type = st.selectbox(
                f"Column Type",
                ["String", "Integer", "Float", "Date", "Boolean", "Categorical"],
                index=["String", "Integer", "Float", "Date", "Boolean", "Categorical"].index(col["type"]),
                key=f"type_{i}"
            )
            st.session_state.columns[i]["type"] = col_type

        # Additional inputs based on column type
        with col3:
            if col_type == "Integer":
                min_val = st.number_input("Min Value", value=col.get("min", 0), key=f"min_{i}")
                max_val = st.number_input("Max Value", value=col.get("max", 100), key=f"max_{i}")
                st.session_state.columns[i]["min"] = min_val
                st.session_state.columns[i]["max"] = max_val

            elif col_type == "Float":
                min_val = st.number_input("Min Value", value=col.get("min", 0.0), step=0.1, key=f"min_{i}")
                max_val = st.number_input("Max Value", value=col.get("max", 100.0), step=0.1, key=f"max_{i}")
                decimals = st.number_input("Decimals", min_value=1, max_value=10, value=col.get("decimals", 2),
                                           key=f"decimals_{i}")
                st.session_state.columns[i]["min"] = min_val
                st.session_state.columns[i]["max"] = max_val
                st.session_state.columns[i]["decimals"] = decimals

            elif col_type == "Date":
                start_date = st.date_input("Start Date", value=col.get("start_date",
                                                                       datetime.date.today() - datetime.timedelta(
                                                                           days=365)), key=f"start_{i}")
                end_date = st.date_input("End Date", value=col.get("end_date", datetime.date.today()),key=f"end_{i}")
                st.session_state.columns[i]["start_date"] = start_date
                st.session_state.columns[i]["end_date"] = end_date

            elif col_type == "Categorical":
                categories_str = st.text_input("Categories (comma-separated)", value=",".join(
                    col.get("categories", ["Category A", "Category B", "Category C"])), key=f"cat_{i}")
                categories = [cat.strip() for cat in categories_str.split(",") if cat.strip()]
                st.session_state.columns[i]["categories"] = categories

            # Display rule input if rules are enabled for this column
            if col.get("has_rule", False):
                rule = st.text_area(
                    "Rule (e.g., 'If age > 18 then is_adult = True' or 'salary should be proportional to experience')",
                    value=col.get("rule", ""),
                    key=f"rule_{i}",
                    help="Specify conditions based on other columns or logical rules for generating this data"
                )
                st.session_state.columns[i]["rule"] = rule

        with col4:
            # Rule toggle button
            rule_label = "✓ Rule" if col.get("has_rule", False) else "➕ Rule"
            rule_style = "primary" if col.get("has_rule", False) else "secondary"
            st.button(rule_label, key=f"toggle_rule_{i}", on_click=toggle_rule, args=(i,), type=rule_style)

        with col5:
            if len(st.session_state.columns) > 1:  # Only show remove button if there's more than one column
                st.button("❌", key=f"remove_{i}", on_click=remove_column, args=(i,))

    # Add column button
    st.button("➕ Add Column", on_click=add_column)

# Generate data button
if st.button("Generate Data"):
    with st.spinner("Generating synthetic data..."):
        try:
            df = generate_data(st.session_state.columns, num_rows)
            st.session_state.generated_df = df
            st.success(f"Successfully generated {len(df)} rows of synthetic data!")
        except Exception as e:
            st.error(f"Error generating data: {e}")

# Display generated data if it exists
if 'generated_df' in st.session_state:
    st.subheader("Generated Data")
    st.dataframe(st.session_state.generated_df)

    # Download button
    csv = convert_df_to_csv(st.session_state.generated_df)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="synthetic_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("Synthetic Data Generator powered by Lakshman Kodela")

