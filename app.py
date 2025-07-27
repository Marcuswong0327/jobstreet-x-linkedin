import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import io
from collections import Counter

def main():
    st.title("JobStreet & LinkedIn Company Data Matcher")
    st.markdown("Upload JobStreet and LinkedIn Excel files to match company data and adjust JobStreet records based on LinkedIn stakeholder counts.")
    
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
            key="jobstreet_upload"
        )
        
        if jobstreet_file is not None:
            try:
                if jobstreet_file.name.endswith('.csv'):
                    st.session_state.jobstreet_data = pd.read_csv(jobstreet_file)
                else:
                    st.session_state.jobstreet_data = pd.read_excel(jobstreet_file)
                
                st.success(f"JobStreet file loaded: {len(st.session_state.jobstreet_data)} rows")
                
                # Validate required columns
                required_cols = ['Job Title', 'Company', 'Location']
                missing_cols = [col for col in required_cols if col not in st.session_state.jobstreet_data.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.info("Required columns: Job Title, Company, Location")
                else:
                    st.success("All required columns found!")
                    
            except Exception as e:
                st.error(f"Error loading JobStreet file: {str(e)}")
    
    with col2:
        st.subheader("LinkedIn Data")
        linkedin_file = st.file_uploader(
            "Upload LinkedIn Excel file",
            type=['xlsx', 'xls'],
            key="linkedin_upload"
        )
        
        if linkedin_file is not None:
            try:
                st.session_state.linkedin_data = pd.read_excel(linkedin_file)
                st.success(f"LinkedIn file loaded: {len(st.session_state.linkedin_data)} rows")
                
                # Validate Current Company column
                if 'Current Company' not in st.session_state.linkedin_data.columns:
                    st.error("Missing 'Current Company' column in LinkedIn data")
                    st.info("LinkedIn file must contain a 'Current Company' column")
                else:
                    st.success("Current Company column found!")
                    
            except Exception as e:
                st.error(f"Error loading LinkedIn file: {str(e)}")
    
    # Process data section
    if st.session_state.jobstreet_data is not None and st.session_state.linkedin_data is not None:
        # Check if both files have required columns
        jobstreet_valid = all(col in st.session_state.jobstreet_data.columns for col in ['Job Title', 'Company', 'Location'])
        linkedin_valid = 'Current Company' in st.session_state.linkedin_data.columns
        
        if jobstreet_valid and linkedin_valid:
            # Center the button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Mix n Match", type="primary", use_container_width=True):
                    with st.spinner("Processing data..."):
                        # Extract company data
                        jobstreet_companies = extract_jobstreet_companies(st.session_state.jobstreet_data)
                        linkedin_companies = extract_linkedin_companies(st.session_state.linkedin_data)
                        
                        # Match companies
                        matches = match_companies(jobstreet_companies, linkedin_companies)
                        
                        # Process the data
                        st.session_state.processed_data = process_jobstreet_data(
                            st.session_state.jobstreet_data, 
                            matches, 
                            linkedin_companies
                        )
                        
                        # Remove any timestamp columns for clean export
                        export_data = st.session_state.processed_data.copy()
                        timestamp_cols = [col for col in export_data.columns if 'extracted' in col.lower() or 'timestamp' in col.lower()]
                        if timestamp_cols:
                            export_data = export_data.drop(columns=timestamp_cols)
                        
                        # Create Excel file and auto-download
                        @st.cache_data
                        def convert_df_to_excel(df):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Processed_JobStreet_Data')
                            processed_data = output.getvalue()
                            return processed_data
                        
                        excel_data = convert_df_to_excel(export_data)
                        
                        # Show success message
                        original_rows = len(st.session_state.jobstreet_data)
                        processed_rows = len(st.session_state.processed_data)
                        added_rows = processed_rows - original_rows
                        
                        st.success(f"Processing completed! Added {added_rows} blank rows. File ready for download.")
                        
                        # Auto-download using JavaScript
                        st.download_button(
                            label="📥 Download Processed Excel File",
                            data=excel_data,
                            file_name="processed_jobstreet_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="auto_download"
                        )

def extract_jobstreet_companies(df):
    """Extract unique company names and their job counts from JobStreet data"""
    if 'Company' not in df.columns:
        return {}
    
    # Clean and count companies
    companies = df['Company'].dropna().str.strip()
    companies = companies[companies != '']  # Remove empty strings
    company_counts = companies.value_counts().to_dict()
    
    return company_counts

def extract_linkedin_companies(df):
    """Extract unique company names and their stakeholder counts from LinkedIn data"""
    if 'Current Company' not in df.columns:
        return {}
    
    # Clean and count companies
    companies = df['Current Company'].dropna().str.strip()
    companies = companies[companies != '']  # Remove empty strings
    company_counts = companies.value_counts().to_dict()
    
    return company_counts

def match_companies(jobstreet_companies, linkedin_companies, threshold=70):
    """Match companies between JobStreet and LinkedIn using fuzzy matching"""
    matches = {}
    
    for jobstreet_company in jobstreet_companies.keys():
        # Find best match in LinkedIn companies
        best_match = process.extractOne(
            jobstreet_company, 
            linkedin_companies.keys(),
            scorer=fuzz.ratio
        )
        
        if best_match and best_match[1] >= threshold:
            matches[jobstreet_company] = (best_match[0], best_match[1])
    
    return matches

def process_jobstreet_data(jobstreet_df, matches, linkedin_companies):
    """Process JobStreet data by adding blank rows based on LinkedIn stakeholder counts"""
    
    # Create a copy of the original data
    processed_df = jobstreet_df.copy()
    
    # Remove any existing timestamp columns
    timestamp_cols = [col for col in processed_df.columns if 'extracted' in col.lower() or 'timestamp' in col.lower()]
    if timestamp_cols:
        processed_df = processed_df.drop(columns=timestamp_cols)
    
    # Group companies and process them one by one
    result_rows = []
    
    # Get unique companies in the original order they appear
    companies_processed = set()
    
    for index, row in processed_df.iterrows():
        company = row['Company']
        
        if company not in companies_processed:
            companies_processed.add(company)
            
            # Get all rows for this company
            company_rows = processed_df[processed_df['Company'] == company].copy()
            
            # Add the original rows
            for _, company_row in company_rows.iterrows():
                result_rows.append(company_row)
            
            # Check if this company has a match in LinkedIn data
            if company in matches:
                linkedin_company, _ = matches[company]
                stakeholder_count = linkedin_companies[linkedin_company]
                
                # Calculate how many blank rows needed
                existing_rows = len(company_rows)
                total_rows_needed = stakeholder_count
                blank_rows_needed = max(0, total_rows_needed - existing_rows)
                
                # Add blank rows if needed
                for i in range(blank_rows_needed):
                    blank_row = pd.Series(index=processed_df.columns, dtype=object)
                    # Keep company name but clear other fields
                    blank_row['Company'] = company
                    blank_row['Job Title'] = ''
                    blank_row['Location'] = ''
                    
                    result_rows.append(blank_row)
    
    # Create new dataframe from result rows
    if result_rows:
        processed_df = pd.DataFrame(result_rows).reset_index(drop=True)
    else:
        processed_df = processed_df.iloc[0:0].copy()  # Empty dataframe with same columns
    
    return processed_df

if __name__ == "__main__":
    main()
