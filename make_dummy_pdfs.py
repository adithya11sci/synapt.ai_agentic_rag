from fpdf import FPDF
from pathlib import Path

def make_dummy_pdfs():
    pdf_dir = Path("data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Infosys PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="INFOSYS FY24 ANNUAL REPORT", ln=True, align='C')
    pdf.cell(200, 10, txt="MD&A", ln=True)
    infosys_text = (
        "Project Maximus five pillars margin expansion are critical to our strategy. "
        "CEO mentioned that Generative AI will transform our delivery. "
        "The main reason for revenue growth FY24 was large deal wins and expansion of existing accounts. "
        "Strategic priorities involve cloud-first approach and AI-first engineering. "
        "We successfully added 100 new customers this year. "
        "Our capital allocation policy is to return 85% of free cash flow to shareholders over a 5-year period. "
        "Workforce strategy focuses on continuous upskilling. net profit reached record highs due to operational efficiency."
    )
    pdf.multi_cell(0, 10, txt=infosys_text)
    pdf.output("data/pdfs/infosys_fy24.pdf")

    # TCS PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="TCS FY24 ANNUAL REPORT", ln=True, align='C')
    pdf.cell(200, 10, txt="CEO Message", ln=True)
    tcs_text = "Strong growth and operating margins driven by AI and cloud priorities. Generative AI integration continues."
    pdf.multi_cell(0, 10, txt=tcs_text)
    pdf.output("data/pdfs/tcs_fy24.pdf")

    # Wipro PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="WIPRO FY24 ANNUAL REPORT", ln=True, align='C')
    pdf.cell(200, 10, txt="Financial Overview", ln=True)
    wipro_text = "Revenue trend shows restructuring efforts. Focus on automation and engineering."
    pdf.multi_cell(0, 10, txt=wipro_text)
    pdf.output("data/pdfs/wipro_fy24.pdf")

    print("Dummy PDFs created!")

if __name__ == "__main__":
    make_dummy_pdfs()
