import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from collections import Counter

# Try to import optional dependencies
try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

if not FUZZYWUZZY_AVAILABLE and not SENTENCE_TRANSFORMERS_AVAILABLE:
    st.warning("⚠️ Advanced matching libraries not available. Using basic string matching.")


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


def apply_duplicate_blanking(df):
    """
    Apply duplicate blanking logic for consecutive rows with identical job title, company, and location.
    Only the first occurrence in each group will show the values, subsequent rows will be blank.
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Columns to apply blanking logic
    blank_columns = ['Job Title', 'Company', 'Location']
    
    # Ensure all columns exist
    for col in blank_columns:
        if col not in result_df.columns:
            continue
    
    # Create a grouping key based on job title, company, and location
    grouping_columns = [col for col in blank_columns if col in result_df.columns]
    
    if not grouping_columns:
        return result_df
    
    # Fill NaN values temporarily for grouping (will restore later)
    temp_df = result_df.copy()
    for col in grouping_columns:
        temp_df[col] = temp_df[col].fillna('__TEMP_NAN__')
    
    # Create groups based on consecutive identical combinations
    current_group = None
    group_start_idx = None
    
    for idx in range(len(temp_df)):
        # Create current row identifier
        current_values = tuple(temp_df.iloc[idx][col] for col in grouping_columns)
        
        if current_group is None:
            # First row - always keep values
            current_group = current_values
            group_start_idx = idx
        elif current_values == current_group:
            # Same group - blank the values for this row
            for col in grouping_columns:
                if col in result_df.columns:
                    result_df.at[result_df.index[idx], col] = ''
        else:
            # New group - keep values and update tracking
            current_group = current_values
            group_start_idx = idx
    
    # Restore original NaN values where they existed
    for col in grouping_columns:
        if col in result_df.columns:
            # Find original NaN positions
            original_nans = df[col].isna()
            # Set them back to NaN in result (but only for first occurrence in each group)
            for idx in range(len(result_df)):
                if original_nans.iloc[idx] and result_df.iloc[idx][col] == '__TEMP_NAN__':
                    result_df.at[result_df.index[idx], col] = None
    
    return result_df


def normalize_company_name(company_name):
    """Enhanced company name normalization for better semantic matching"""
    if pd.isna(company_name) or company_name == '':
        return ''
    
    # Convert to string and strip whitespace
    name = str(company_name).strip()
    
    # Remove common company suffixes and variations (more comprehensive)
    suffixes_to_remove = [
        r'\s+Pty\s+Ltd\.?$', r'\s+Pty\.?\s+Ltd\.?$', r'\s+PTY\s+LTD\.?$',
        r'\s+Ltd\.?$', r'\s+LTD\.?$', r'\s+Limited\.?$', r'\s+LIMITED\.?$',
        r'\s+Inc\.?$', r'\s+INC\.?$', r'\s+Incorporated\.?$', r'\s+INCORPORATED\.?$',
        r'\s+Corp\.?$', r'\s+CORP\.?$', r'\s+Corporation\.?$', r'\s+CORPORATION\.?$',
        r'\s+Co\.?$', r'\s+CO\.?$', r'\s+Company\.?$', r'\s+COMPANY\.?$',
        r'\s+LLC\.?$', r'\s+L\.L\.C\.?$', r'\s+LLP\.?$', r'\s+L\.L\.P\.?$',
        r'\s+Group\.?$', r'\s+GROUP\.?$', r'\s+Holdings\.?$', r'\s+HOLDINGS\.?$',
        r'\s+International\.?$', r'\s+INTERNATIONAL\.?$', r'\s+Global\.?$', r'\s+GLOBAL\.?$'
    ]
    
    # Apply suffix removal
    for suffix_pattern in suffixes_to_remove:
        name = re.sub(suffix_pattern, '', name, flags=re.IGNORECASE)
    
    # Remove common abbreviations and standardize
    abbreviation_replacements = {
        r'\b&\b': 'and',
        r'\btech\b': 'technology',
        r'\bsystems?\b': 'system',
        r'\benergy\b': '',  # Make energy optional for matching
        r'\bsolutions?\b': '',  # Make solutions optional for matching
        r'\bindustries\b': '',  # Make industries optional for matching
        r'\btechnologies\b': 'technology',
        r'\bservices?\b': '',  # Make services optional for matching
    }
    
    # Apply abbreviation replacements
    for pattern, replacement in abbreviation_replacements.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    
    # Remove extra whitespace, punctuation, and normalize case
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = ' '.join(name.split()).strip().lower()
    
    return name


def extract_core_company_name(company_name):
    """Extract the core company name for advanced semantic matching"""
    if pd.isna(company_name) or company_name == '':
        return ''
    
    name = str(company_name).strip()
    
    # Handle specific patterns like "Techtronic Industries - TTI"
    if ' - ' in name:
        parts = name.split(' - ')
        # Take the part that's likely the main name (usually the longer one)
        name = max(parts, key=len).strip()
    
    # Handle patterns with parentheses
    name = re.sub(r'\([^)]*\)', '', name).strip()
    
    # Apply basic normalization
    name = normalize_company_name(name)
    
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


# Cache the model to avoid re-loading it every time the app re-runs
@st.cache_resource
def get_sentence_transformer():
    """Load and cache the Sentence-Transformer model."""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    else:
        return None


def match_companies_semantic(jobstreet_companies, linkedin_companies, threshold_score=0.75):
    """
    Enhanced semantic company matching with multiple normalization approaches.
    
    Args:
        jobstreet_companies (dict): A dictionary of company names from JobStreet.
        linkedin_companies (dict): A dictionary of company names from LinkedIn.
        threshold_score (float): The minimum semantic similarity score (0-1) to consider a match.
        
    Returns:
        dict: A dictionary mapping JobStreet company names to a tuple of 
              (LinkedIn company name, score).
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return {}
        
    model = get_sentence_transformer()
    if model is None:
        return {}
    
    # Get unique company names and ensure they are not empty
    js_names = [name for name in jobstreet_companies.keys() if name and isinstance(name, str)]
    li_names = [name for name in linkedin_companies.keys() if name and isinstance(name, str)]
    
    if not js_names or not li_names:
        return {}

    matches = {}
    
    # Try multiple matching approaches with different normalization levels
    for js_name in js_names:
        best_match = None
        best_score = 0
        
        # Approach 1: Raw company names
        js_embedding = model.encode([js_name], convert_to_tensor=True)
        li_embeddings = model.encode(li_names, convert_to_tensor=True)
        cosine_scores = util.cos_sim(js_embedding, li_embeddings)[0]
        
        for j, li_name in enumerate(li_names):
            score = cosine_scores[j].item()
            if score > best_score:
                best_score = score
                best_match = li_name
        
        # Approach 2: Normalized company names
        js_normalized = normalize_company_name(js_name)
        if js_normalized:
            li_normalized = [normalize_company_name(name) for name in li_names]
            li_normalized = [name for name in li_normalized if name]  # Remove empty strings
            
            if li_normalized:
                js_norm_embedding = model.encode([js_normalized], convert_to_tensor=True)
                li_norm_embeddings = model.encode(li_normalized, convert_to_tensor=True)
                norm_cosine_scores = util.cos_sim(js_norm_embedding, li_norm_embeddings)[0]
                
                for j, (norm_name, orig_name) in enumerate(zip(li_normalized, li_names)):
                    if j < len(norm_cosine_scores):
                        score = norm_cosine_scores[j].item()
                        if score > best_score:
                            best_score = score
                            best_match = orig_name
        
        # Approach 3: Core company names (handles complex cases like "Techtronic Industries - TTI")
        js_core = extract_core_company_name(js_name)
        if js_core:
            li_core = [extract_core_company_name(name) for name in li_names]
            li_core = [name for name in li_core if name]  # Remove empty strings
            
            if li_core:
                js_core_embedding = model.encode([js_core], convert_to_tensor=True)
                li_core_embeddings = model.encode(li_core, convert_to_tensor=True)
                core_cosine_scores = util.cos_sim(js_core_embedding, li_core_embeddings)[0]
                
                for j, (core_name, orig_name) in enumerate(zip(li_core, li_names)):
                    if j < len(core_cosine_scores):
                        score = core_cosine_scores[j].item()
                        if score > best_score:
                            best_score = score
                            best_match = orig_name
        
        # Record match if it exceeds threshold
        if best_score >= threshold_score and best_match:
            matches[js_name] = (best_match, round(best_score * 100))
            
    return matches


def get_linkedin_employees_for_company(linkedin_df, company_name):
    """Get all LinkedIn employees for a specific company"""
    if 'Current Company' not in linkedin_df.columns:
        return pd.DataFrame()
    
    # Filter employees for the specific company
    return linkedin_df[linkedin_df['Current Company'] == company_name].copy()


def match_companies_enhanced(jobstreet_companies, linkedin_companies, threshold=75):
    """
    Enhanced company matching using available matching methods
    """
    matches = {}
    
    # First try semantic matching if available
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        semantic_threshold = threshold / 100.0
        semantic_matches = match_companies_semantic(jobstreet_companies, linkedin_companies, semantic_threshold)
        matches.update(semantic_matches)
    
    # For unmatched companies, try fuzzy matching if available
    unmatched_js = [company for company in jobstreet_companies.keys() if company not in matches]
    
    if FUZZYWUZZY_AVAILABLE and unmatched_js:
        for js_company in unmatched_js:
            # Normalize company names for better fuzzy matching
            normalized_js = normalize_company_name(js_company)
            
            best_match = None
            best_score = 0
            
            for li_company in linkedin_companies.keys():
                normalized_li = normalize_company_name(li_company)
                
                # Try different fuzzy matching approaches
                scores = [
                    fuzz.ratio(normalized_js.lower(), normalized_li.lower()),
                    fuzz.partial_ratio(normalized_js.lower(), normalized_li.lower()),
                    fuzz.token_sort_ratio(normalized_js.lower(), normalized_li.lower()),
                    fuzz.token_set_ratio(normalized_js.lower(), normalized_li.lower())
                ]
                
                max_score = max(scores)
                
                if max_score > best_score and max_score >= threshold:
                    best_score = max_score
                    best_match = li_company
            
            if best_match:
                matches[js_company] = (best_match, best_score)
    elif not FUZZYWUZZY_AVAILABLE and unmatched_js:
        # Fallback to basic string matching
        for js_company in unmatched_js:
            normalized_js = normalize_company_name(js_company).lower()
            
            for li_company in linkedin_companies.keys():
                normalized_li = normalize_company_name(li_company).lower()
                
                # Simple exact match after normalization
                if normalized_js == normalized_li and normalized_js:
                    matches[js_company] = (li_company, 100)
                    break
                # Simple substring match
                elif normalized_js in normalized_li or normalized_li in normalized_js:
                    if len(normalized_js) > 3 and len(normalized_li) > 3:  # Avoid very short matches
                        matches[js_company] = (li_company, 80)
    
    return matches


def process_jobstreet_data_enhanced(jobstreet_df, linkedin_df, company_matches, linkedin_companies):
    """
    Process JobStreet data and create expanded dataset with LinkedIn employee information
    """
    result_rows = []
    
    for _, js_row in jobstreet_df.iterrows():
        js_company = js_row['Company']
        
        # Check if this company has a match
        if js_company in company_matches:
            matched_li_company, match_score = company_matches[js_company]
            
            # Get all employees for this LinkedIn company
            employees = get_linkedin_employees_for_company(linkedin_df, matched_li_company)
            
            if not employees.empty:
                # Create a row for each employee
                for _, employee in employees.iterrows():
                    result_row = {
                        'Job Title': js_row['Job Title'],
                        'Company': js_row['Company'],
                        'Location': js_row['Location'],
                        'Name': employee.get('Name', ''),
                        'First Name': employee.get('First Name', ''),
                        "Stakeholder's position": employee.get('Current Role', ''),
                        'Email': employee.get('Email', ''),
                        'Match Score': match_score,
                        'LinkedIn Company': matched_li_company
                    }
                    result_rows.append(result_row)
            else:
                # No employees found, add original row with empty employee fields
                result_row = {
                    'Job Title': js_row['Job Title'],
                    'Company': js_row['Company'],
                    'Location': js_row['Location'],
                    'Name': '',
                    'First Name': '',
                    "Stakeholder's position": '',
                    'Email': '',
                    'Match Score': match_score,
                    'LinkedIn Company': matched_li_company
                }
                result_rows.append(result_row)
        else:
            # No company match found, add original row with empty employee fields
            result_row = {
                'Job Title': js_row['Job Title'],
                'Company': js_row['Company'],
                'Location': js_row['Location'],
                'Name': '',
                'First Name': '',
                "Stakeholder's position": '',
                'Email': '',
                'Match Score': 0,
                'LinkedIn Company': ''
            }
            result_rows.append(result_row)
    
    # Create DataFrame and apply duplicate blanking
    processed_df = pd.DataFrame(result_rows)
    return apply_duplicate_blanking(processed_df)


def convert_df_to_excel(df):
    """Convert DataFrame to Excel bytes for download"""
    output = io.BytesIO()
    
    # Create the final output with only required columns
    final_columns = ['Job Title', 'Company', 'Location', 'Name', 'First Name', "Stakeholder's position", 'Email']
    final_df = df[final_columns].copy()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Processed Data')
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Processed Data']
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
    
    output.seek(0)
    return output.getvalue()


if __name__ == "__main__":
    main()
