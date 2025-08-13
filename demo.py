import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Streamlit App Main Function
# ----------------------------
def main():
    st.title("JobStreet & LinkedIn Company Data Matcher (Semantic Matching)")
    st.markdown(
        "Upload JobStreet and LinkedIn Excel files to match company data and map employee details "
        "with intelligent company matching using **semantic similarity**."
    )

    # Initialize session state
    if 'jobstreet_data' not in st.session_state:
        st.session_state.jobstreet_data = None
    if 'linkedin_data' not in st.session_state:
        st.session_state.linkedin_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # File upload section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("JobStreet Data")
        jobstreet_file = st.file_uploader(
            "Upload JobStreet CSV/Excel file",
            type=['csv', 'xlsx', 'xls'],
            key="jobstreet_upload",
            help="Required columns: Job Title, Company, Location"
        )

        if jobstreet_file is not None:
            try:
                if jobstreet_file.name.endswith('.csv'):
                    st.session_state.jobstreet_data = pd.read_csv(jobstreet_file)
                else:
                    st.session_state.jobstreet_data = pd.read_excel(jobstreet_file)

                st.success(f"JobStreet file loaded: {len(st.session_state.jobstreet_data)} rows")

                required_cols = ['Job Title', 'Company', 'Location']
                missing_cols = [col for col in required_cols if col not in st.session_state.jobstreet_data.columns]

                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    st.success("All required columns found!")
                    with st.expander("Preview JobStreet Data"):
                        st.dataframe(st.session_state.jobstreet_data.head())

            except Exception as e:
                st.error(f"Error loading JobStreet file: {str(e)}")

    with col2:
        st.subheader("LinkedIn Data")
        linkedin_file = st.file_uploader(
            "Upload LinkedIn Excel file",
            type=['xlsx', 'xls', 'csv'],
            key="linkedin_upload",
            help="Required columns: Name, First Name, Last Name, Email, Current Role, Current Company"
        )

        if linkedin_file is not None:
            try:
                if linkedin_file.name.endswith('.csv'):
                    st.session_state.linkedin_data = pd.read_csv(linkedin_file)
                else:
                    st.session_state.linkedin_data = pd.read_excel(linkedin_file)

                st.success(f"LinkedIn file loaded: {len(st.session_state.linkedin_data)} rows")

                linkedin_required_cols = ['Name', 'First Name', 'Last Name', 'Email', 'Current Role', 'Current Company']
                linkedin_missing_cols = [col for col in linkedin_required_cols if col not in st.session_state.linkedin_data.columns]

                if linkedin_missing_cols:
                    st.error(f"Missing required columns: {linkedin_missing_cols}")
                else:
                    st.success("All required columns found!")
                    with st.expander("Preview LinkedIn Data"):
                        st.dataframe(st.session_state.linkedin_data.head())

            except Exception as e:
                st.error(f"Error loading LinkedIn file: {str(e)}")

    # Process data section
    if st.session_state.jobstreet_data is not None and st.session_state.linkedin_data is not None:
        jobstreet_valid = all(col in st.session_state.jobstreet_data.columns for col in ['Job Title', 'Company', 'Location'])
        linkedin_valid = all(col in st.session_state.linkedin_data.columns for col in ['Name', 'First Name', 'Last Name', 'Email', 'Current Role', 'Current Company'])

        if jobstreet_valid and linkedin_valid:
            st.divider()
            st.subheader("Matching Configuration")

            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider(
                    "Company Matching Threshold (Semantic Similarity)",
                    0.50, 1.00, 0.75, 0.01,
                    help="Higher values require more exact matches"
                )
            with col2:
                preview_matches = st.checkbox("Preview company matches before processing", value=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Process & Match Data", type="primary", use_container_width=True):
                    with st.spinner("Processing data..."):
                        jobstreet_companies = extract_jobstreet_companies(st.session_state.jobstreet_data)
                        linkedin_companies = extract_linkedin_companies(st.session_state.linkedin_data)

                        matches = match_companies_semantic(jobstreet_companies, linkedin_companies, threshold)

                        if preview_matches and matches:
                            st.subheader("Company Matches Found")
                            match_df = pd.DataFrame([
                                {
                                    'JobStreet Company': js_company,
                                    'LinkedIn Company': li_company,
                                    'Match Score (%)': score,
                                    'LinkedIn Employees': linkedin_companies[li_company]
                                }
                                for js_company, (li_company, score) in matches.items()
                            ])
                            st.dataframe(match_df, use_container_width=True)

                        st.session_state.processed_data = process_jobstreet_data_enhanced(
                            st.session_state.jobstreet_data,
                            st.session_state.linkedin_data,
                            matches,
                            linkedin_companies
                        )

                        excel_data = convert_df_to_excel(st.session_state.processed_data)

                        original_rows = len(st.session_state.jobstreet_data)
                        processed_rows = len(st.session_state.processed_data)
                        added_rows = processed_rows - original_rows

                        st.success(f"✅ Processing completed! Added {added_rows} rows with employee data mapping.")

                        st.download_button(
                            label="📥 Download Processed Excel File",
                            data=excel_data,
                            file_name="processed_jobstreet_linkedin_semantic.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        with st.expander("Preview Processed Data"):
                            st.dataframe(st.session_state.processed_data.head(20))


# ----------------------------
# Helper Functions
# ----------------------------

def extract_jobstreet_companies(df):
    if 'Company' not in df.columns:
        return {}
    companies = df['Company'].dropna().str.strip()
    companies = companies[companies != '']
    return companies.value_counts().to_dict()

def extract_linkedin_companies(df):
    if 'Current Company' not in df.columns:
        return {}
    companies = df['Current Company'].dropna().str.strip()
    companies = companies[companies != '']
    return companies.value_counts().to_dict()

@st.cache_resource
def get_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

def match_companies_semantic(jobstreet_companies, linkedin_companies, threshold_score=0.75):
    model = get_sentence_transformer()

    js_names = [name for name in jobstreet_companies.keys() if name and isinstance(name, str)]
    li_names = [name for name in linkedin_companies.keys() if name and isinstance(name, str)]

    if not js_names or not li_names:
        return {}

    js_embeddings = model.encode(js_names, convert_to_tensor=True)
    li_embeddings = model.encode(li_names, convert_to_tensor=True)

    matches = {}
    cosine_scores = util.cos_sim(js_embeddings, li_embeddings)

    for i, js_name in enumerate(js_names):
        best_match_index = int(np.argmax(cosine_scores[i]).item())
        best_match_score = cosine_scores[i][best_match_index].item()
        if best_match_score >= threshold_score:
            matched_li_name = li_names[best_match_index]
            matches[js_name] = (matched_li_name, round(best_match_score * 100))
    return matches

def get_linkedin_employees_for_company(linkedin_df, company_name):
    if 'Current Company' not in linkedin_df.columns:
        return pd.DataFrame()
    company_employees = linkedin_df[linkedin_df['Current Company'] == company_name].copy()
    company_employees = company_employees.dropna(subset=['First Name', 'Current Role'])
    return company_employees

def process_jobstreet_data_enhanced(jobstreet_df, linkedin_df, matches, linkedin_companies):
    processed_df = jobstreet_df.copy()
    processed_df['First Name'] = ''
    processed_df['Title'] = ''
    processed_df['Email'] = ''

    timestamp_cols = [col for col in processed_df.columns if 'extracted' in col.lower() or 'timestamp' in col.lower()]
    if timestamp_cols:
        processed_df = processed_df.drop(columns=timestamp_cols)

    result_rows = []
    companies_processed = set()

    for company in processed_df['Company'].unique():
        if company in companies_processed:
            continue
        companies_processed.add(company)

        company_rows = processed_df[processed_df['Company'] == company].copy()

        if company in matches:
            linkedin_company, _ = matches[company]
            linkedin_employees = get_linkedin_employees_for_company(linkedin_df, linkedin_company)

            if not linkedin_employees.empty:
                employee_list = linkedin_employees.to_dict('records')

                # Fill existing rows
                for i, (_, row) in enumerate(company_rows.iterrows()):
                    row_to_add = row.copy()
                    if i < len(employee_list):
                        employee = employee_list[i]
                        row_to_add['First Name'] = employee.get('First Name', '')
