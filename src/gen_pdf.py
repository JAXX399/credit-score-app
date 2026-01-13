from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Loan Application Form")

    c.setFont("Helvetica", 12)
    text_content = [
        "Applicant Name: Max Mustermann",
        "Age: 35 years old",
        "Gender: Male (Single)",
        "Nationality: German (Foreign worker: No)",
        "",
        "--- Financial Details ---",
        "Checking Status: no checking account",
        "Savings Balance: 800 DM (Savings)",
        "Job: Skilled employee",
        "Employment: 5 years at current company",
        "Number of Dependents: 1",
        "",
        "--- Loan Request ---",
        "Purpose: New Car purchase",
        "Credit Amount: 5000 DM",
        "Duration: 24 months",
        "Installment Rate: 2% of disposable income",
        "",
        "--- Assets & Living ---",
        "Housing: Own house",
        "Property: Real Estate",
        "Existing Credits: 1"
    ]

    y_position = height - 100
    for line in text_content:
        c.drawString(50, y_position, line)
        y_position -= 20

    c.save()
    print(f"PDF created: {filename}")

if __name__ == "__main__":
    create_pdf("dummy_application.pdf")
