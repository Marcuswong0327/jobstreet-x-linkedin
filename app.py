import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import io
import re
from collections import Counter

def main():
    st.title("JobStreet & LinkedIn Company Data Matcher")
    st.markdown("Upload JobStreet and LinkedIn Excel files to match company data and map employee details with intelligent company matching.")
    
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
                
                # Validate required columns
                required_cols = ['Job Title', 'Company', 'Location']
                missing_cols = [col for col in required_cols if col not in st.session_state.jobstreet_data.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.info("Required columns: Job Title, Company, Location")
                else:
                    st.success("All required columns found!")
                    # Show preview
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
                
                # Validate required columns
                linkedin_required_cols = ['Name', 'First Name', 'Last Name', 'Email', 'Current Role', 'Current Company']
                linkedin_missing_cols = [col for col in linkedin_required_cols if col not in st.session_state.linkedin_data.columns]
                
                if linkedin_missing_cols:
                    st.error(f"Missing required columns: {linkedin_missing_cols}")
                    st.info("Required columns: Name, First Name, Last Name, Email, Current Role, Current Company")
                else:
                    st.success("All required columns found!")
                    # Show preview
                    with st.expander("Preview LinkedIn Data"):
                        st.dataframe(st.session_state.linkedin_data.head())
                    
            except Exception as e:
                st.error(f"Error loading LinkedIn file: {str(e)}")
    
    # Process data section
    if st.session_state.jobstreet_data is not None and st.session_state.linkedin_data is not None:
        # Check if both files have required columns
        jobstreet_valid = all(col in st.session_state.jobstreet_data.columns for col in ['Job Title', 'Company', 'Location'])
        linkedin_valid = all(col in st.session_state.linkedin_data.columns for col in ['Name', 'First Name', 'Last Name', 'Email', 'Current Role', 'Current Company'])
        
        if jobstreet_valid and linkedin_valid:
            st.divider()
            
            # Matching settings
            st.subheader("Matching Configuration")
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Company Matching Threshold", 50, 100, 75, 
                                    help="Higher values require more exact matches")
            with col2:
                preview_matches = st.checkbox("Preview company matches before processing", value=True)
            
            # Center the button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Process & Match Data", type="primary", use_container_width=True):
                    with st.spinner("Processing data..."):
                        # Extract company data
                        jobstreet_companies = extract_jobstreet_companies(st.session_state.jobstreet_data)
                        linkedin_companies = extract_linkedin_companies(st.session_state.linkedin_data)
                        
                        # Match companies with enhanced matching
                        matches = match_companies_enhanced(jobstreet_companies, linkedin_companies, threshold)
                        
                        if preview_matches and matches:
                            st.subheader("Company Matches Found")
                            match_df = pd.DataFrame([
                                {
                                    'JobStreet Company': js_company,
                                    'LinkedIn Company': li_company,
                                    'Match Score': score,
                                    'LinkedIn Employees': linkedin_companies[li_company]
                                }
                                for js_company, (li_company, score) in matches.items()
                            ])
                            st.dataframe(match_df, use_container_width=True)
                        
                        # Process the data with employee mapping
                        st.session_state.processed_data = process_jobstreet_data_enhanced(
                            st.session_state.jobstreet_data, 
                            st.session_state.linkedin_data,
                            matches, 
                            linkedin_companies
                        )
                        
                        # Create Excel file for download
                        excel_data = convert_df_to_excel(st.session_state.processed_data)
                        
                        # Show success message
                        original_rows = len(st.session_state.jobstreet_data)
                        processed_rows = len(st.session_state.processed_data)
                        added_rows = processed_rows - original_rows
                        
                        st.success(f"✅ Processing completed! Added {added_rows} rows with employee data mapping.")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Processed Excel File",
                            data=excel_data,
                            file_name="processed_jobstreet_linkedin_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Show preview of results
                        with st.expander("Preview Processed Data"):
                            st.dataframe(st.session_state.processed_data.head(20))

def normalize_company_name(company_name):
    """Normalize company name for better matching"""
    if pd.isna(company_name) or company_name == '':
        return ''
    
    # Convert to string and strip whitespace
    name = str(company_name).strip()
    
    # Remove common company suffixes and variations
    suffixes_to_remove = [
        r'\s+Pty\s+Ltd\.?$', r'\s+Pty\.?\s+Ltd\.?$', r'\s+PTY\s+LTD\.?$',
        r'\s+Ltd\.?$', r'\s+LTD\.?$', r'\s+Limited\.?$', r'\s+LIMITED\.?$',
        r'\s+Inc\.?$', r'\s+INC\.?$', r'\s+Incorporated\.?$', r'\s+INCORPORATED\.?$',
        r'\s+Corp\.?$', r'\s+CORP\.?$', r'\s+Corporation\.?$', r'\s+CORPORATION\.?$',
        r'\s+Co\.?$', r'\s+CO\.?$', r'\s+Company\.?$', r'\s+COMPANY\.?$',
        r'\s+LLC\.?$', r'\s+L\.L\.C\.?$', r'\s+LLP\.?$', r'\s+L\.L\.P\.?$'
    ]
    
    # Apply suffix removal
    for suffix_pattern in suffixes_to_remove:
        name = re.sub(suffix_pattern, '', name, flags=re.IGNORECASE)
    
    # Clean up extra whitespace and normalize case
    name = ' '.join(name.split()).strip()
    
    return name

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

def match_companies_enhanced(jobstreet_companies, linkedin_companies, threshold=75):
    """Enhanced company matching using fuzzy matching with name normalization"""
    matches = {}
    
    # Create normalized mapping for LinkedIn companies
    linkedin_normalized = {}
    for li_company in linkedin_companies.keys():
        normalized = normalize_company_name(li_company)
        if normalized:
            linkedin_normalized[li_company] = normalized
    
    for js_company in jobstreet_companies.keys():
        js_normalized = normalize_company_name(js_company)
        if not js_normalized:
            continue
            
        best_match = None
        best_score = 0
        
        # Try exact normalized match first
        for li_company, li_normalized in linkedin_normalized.items():
            if js_normalized.lower() == li_normalized.lower():
                matches[js_company] = (li_company, 100)
                best_match = li_company
                break
        
        # If no exact match, use fuzzy matching on both original and normalized names
        if not best_match:
            # Test against original company names
            match_result = process.extractOne(
                js_company, 
                linkedin_companies.keys(),
                scorer=fuzz.ratio
            )
            
            if match_result and match_result[1] >= threshold:
                best_match = match_result[0]
                best_score = match_result[1]
            
            # Also test against normalized names
            normalized_match = process.extractOne(
                js_normalized,
                list(linkedin_normalized.values()),
                scorer=fuzz.ratio
            )
            
            if normalized_match and normalized_match[1] > best_score and normalized_match[1] >= threshold:
                # Find the original company name for this normalized match
                for li_company, li_normalized in linkedin_normalized.items():
                    if li_normalized == normalized_match[0]:
                        best_match = li_company
                        best_score = normalized_match[1]
                        break
            
            if best_match:
                matches[js_company] = (best_match, best_score)
    
    return matches

def get_linkedin_employees_for_company(linkedin_df, company_name):
    """Get all LinkedIn employees for a specific company"""
    if 'Current Company' not in linkedin_df.columns:
        return pd.DataFrame()
    
    # Filter employees for the specific company
    company_employees = linkedin_df[linkedin_df['Current Company'] == company_name].copy()
    
    # Clean the data
    company_employees = company_employees.dropna(subset=['First Name', 'Current Role'])
    
    return company_employees

def process_jobstreet_data_enhanced(jobstreet_df, linkedin_df, matches, linkedin_companies):
    """Enhanced processing with employee detail mapping"""
    
    # Create a copy of the original data and add new columns
    processed_df = jobstreet_df.copy()
    
    # Add new columns for employee details
    processed_df['First Name'] = ''
    processed_df['Title'] = ''
    processed_df['Email'] = ''
    
    # Remove any existing timestamp columns
    timestamp_cols = [col for col in processed_df.columns if 'extracted' in col.lower() or 'timestamp' in col.lower()]
    if timestamp_cols:
        processed_df = processed_df.drop(columns=timestamp_cols)
    
    # Group companies and process them one by one
    result_rows = []
    companies_processed = set()
    
    for index, row in processed_df.iterrows():
        company = row['Company']
        
        if company not in companies_processed:
            companies_processed.add(company)
            
            # Get all rows for this company
            company_rows = processed_df[processed_df['Company'] == company].copy()
            
            # Check if this company has a match in LinkedIn data
            if company in matches:
                linkedin_company, match_score = matches[company]
                
                # Get LinkedIn employees for this company
                linkedin_employees = get_linkedin_employees_for_company(linkedin_df, linkedin_company)
                
                if not linkedin_employees.empty:
                    employee_list = linkedin_employees.to_dict('records')
                    
                    # Add original JobStreet rows first, populate with employee data if available
                    for i, (_, company_row) in enumerate(company_rows.iterrows()):
                        row_to_add = company_row.copy()
                        
                        # If we have LinkedIn employee data, populate the first rows
                        if i < len(employee_list):
                            employee = employee_list[i]
                            row_to_add['First Name'] = employee.get('First Name', '')
                            row_to_add['Title'] = employee.get('Current Role', '')
                            row_to_add['Email'] = employee.get('Email', '')
                        
                        result_rows.append(row_to_add)
                    
                    # Calculate additional blank rows needed
                    existing_rows = len(company_rows)
                    total_employees = len(employee_list)
                    blank_rows_needed = max(0, total_employees - existing_rows)
                    
                    # Add blank rows with employee data
                    for i in range(blank_rows_needed):
                        blank_row = pd.Series(index=processed_df.columns, dtype=object)
                        blank_row['Job Title'] = ''
                        blank_row['Company'] = company
                        blank_row['Location'] = ''
                        
                        # Add employee data from LinkedIn
                        employee_index = existing_rows + i
                        if employee_index < len(employee_list):
                            employee = employee_list[employee_index]
                            blank_row['First Name'] = employee.get('First Name', '')
                            blank_row['Title'] = employee.get('Current Role', '')
                            blank_row['Email'] = employee.get('Email', '')
                        else:
                            blank_row['First Name'] = ''
                            blank_row['Title'] = ''
                            blank_row['Email'] = ''
                        
                        result_rows.append(blank_row)
                else:
                    # No LinkedIn employees found, just add original rows
                    for _, company_row in company_rows.iterrows():
                        result_rows.append(company_row)
            else:
                # No match found, just add original rows
                for _, company_row in company_rows.iterrows():
                    result_rows.append(company_row)
    
    # Create new dataframe from result rows
    if result_rows:
        final_df = pd.DataFrame(result_rows).reset_index(drop=True)
    else:
        final_df = processed_df.iloc[0:0].copy()  # Empty dataframe with same columns
    
    return final_df

@st.cache_data
def convert_df_to_excel(df):
    """Convert dataframe to Excel format for download"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed_Data')
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Processed_Data']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    processed_data = output.getvalue()
    return processed_data

if __name__ == "__main__":
    main()
