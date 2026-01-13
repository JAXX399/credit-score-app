import PyPDF2
from docx import Document
import re
import io

def parse_document(uploaded_file):
    """
    Extracts text from PDF or DOCX file.
    """
    text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            # Assume txt
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"
    
    return text

def extract_attributes(text):
    """
    Uses regex and keyword matching to find credit attributes in text.
    Returns a dictionary of found values mapped to frontend keys.
    """
    text = text.lower()
    data = {}

    # --- 1. Personal ---
    # Age
    age_match = re.search(r'age[:\s]+(\d{2})', text)
    if age_match:
        data['age'] = int(age_match.group(1))

    # Sex (Heuristic)
    if 'female' in text or 'woman' in text or 'mrs.' in text or 'ms.' in text:
        # Defaulting to Female (Div/Mar) as it's a common category
        data['sex'] = "Female (Div/Mar)" 
    elif 'male' in text or 'man' in text or 'mr.' in text:
        # Defaulting to Single Male (safest common assumption if not specified)
        data['sex'] = "Male (Single)"

    # Dependents
    dep_match = re.search(r'dependents[:\s]+(\d+)', text)
    if dep_match:
        # Map >1 to 2 (Since UI radio is 1 or 2)
        val = int(dep_match.group(1))
        data['dependents'] = 2 if val > 1 else 1

    # Foreign
    if 'foreign' in text:
        if 'no' in text.split('foreign')[1][:10]:
            data['foreign'] = "No"
        else:
            data['foreign'] = "Yes"

    # Job Type
    if 'unemployed' in text:
        data['job'] = "Unemployed"
    elif 'management' in text or 'director' in text or 'highly qualified' in text:
        data['job'] = "Management"
    elif 'skilled' in text:
        data['job'] = "Skilled"
    elif 'unskilled' in text:
        data['job'] = "Unskilled (Res)"

    # Employment Duration
    # Search for patterns like "5 years", "employ", etc.
    emp_match = re.search(r'employment[:\s]+(\d+)', text)
    if emp_match:
        years = int(emp_match.group(1))
        if years < 1: data['emp_duration'] = "< 1 year"
        elif years < 4: data['emp_duration'] = "1-4 years"
        elif years < 7: data['emp_duration'] = "4-7 years"
        else: data['emp_duration'] = ">= 7 years"

    # --- 2. Financial ---
    # Checking Status
    if 'no checking' in text or 'no account' in text:
        data['check_status'] = "No Account (Safe)"
    elif 'negative' in text or 'overdraft' in text:
        data['check_status'] = "Negative (<0)"
    elif 'checking' in text:
         # Rough heuristic for amounts
         amt = re.search(r'checking[:\s]+(\d+)', text)
         if amt:
             val = int(amt.group(1))
             if val > 200: data['check_status'] = "High (>200)"
             else: data['check_status'] = "Low (0-200)"

    # Savings
    if 'no savings' in text:
        data['savings'] = "Unknown/None"
    elif 'savings' in text:
         amt = re.search(r'savings[:\s]+(\d+)', text)
         if amt:
             val = int(amt.group(1))
             if val < 100: data['savings'] = "Low (<100)"
             elif val < 500: data['savings'] = "Medium"
             elif val < 1000: data['savings'] = "High"
             else: data['savings'] = "Very High"

    # Existing Credits
    cred_match = re.search(r'existing credits[:\s]+(\d+)', text)
    if cred_match:
        data['exist_credits'] = int(cred_match.group(1))

    # --- 3. Assets ---
    # Housing
    if 'own' in text and 'house' in text:
        data['housing'] = "Own"
    elif 'rent' in text:
        data['housing'] = "Rent"
    elif 'free' in text and 'housing' in text:
        data['housing'] = "Free"

    # Property
    if 'real estate' in text:
        data['property'] = "Real Estate"
    elif 'car' in text and 'property' in text:
        data['property'] = "Car/Other"
    elif 'insurance' in text and 'life' in text:
        data['property'] = "Savings/Life Ins"

    # --- 4. Loan Details ---
    # Amount
    amt_match = re.search(r'amount[:\s]+(\d+)', text)
    if amt_match:
        data['amount'] = int(amt_match.group(1))

    # Duration
    dur_match = re.search(r'duration[:\s]+(\d+)', text)
    if dur_match:
        data['duration'] = int(dur_match.group(1))
    
    # Purpose
    if 'new car' in text: data['purpose'] = "New Car"
    elif 'used car' in text: data['purpose'] = "Used Car"
    elif 'furniture' in text: data['purpose'] = "Furniture"
    elif 'radio' in text or 'tv' in text: data['purpose'] = "Radio/TV"
    elif 'education' in text: data['purpose'] = "Education"
    elif 'business' in text: data['purpose'] = "Business"
    elif 'repair' in text: data['purpose'] = "Repairs"

    return data
