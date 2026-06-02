from fpdf import FPDF
import os

def create_sales_report_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Acme Corp - Annual Sales Report 2024", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Subtitle
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 10, "Prepared by: Data Analytics Team | Date: January 2025", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    # Section 1
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8,
        "Acme Corp achieved total revenue of $48.5 million in 2024, representing a 12% "
        "increase compared to $43.3 million in 2023. The company served over 15,000 customers "
        "across 5 regions. The North America region contributed the highest revenue at $22.1 million. "
        "Overall profit margin improved from 18% to 21% driven by cost optimization initiatives."
    )
    pdf.ln(5)

    # Section 2
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Revenue by Region", new_x="LMARGIN", new_y="NEXT")
    regions = [
        ("North America", "$22.1M", "45.6%", "+15%"),
        ("Europe",        "$12.3M", "25.4%", "+10%"),
        ("Asia Pacific",  "$8.7M",  "17.9%", "+18%"),
        ("Latin America", "$3.2M",  "6.6%",  "+5%"),
        ("Middle East",   "$2.2M",  "4.5%",  "+8%"),
    ]
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(60, 8, "Region",     border=1)
    pdf.cell(35, 8, "Revenue",    border=1)
    pdf.cell(35, 8, "Share",      border=1)
    pdf.cell(35, 8, "YoY Growth", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 11)
    for region, revenue, share, growth in regions:
        pdf.cell(60, 8, region,  border=1)
        pdf.cell(35, 8, revenue, border=1)
        pdf.cell(35, 8, share,   border=1)
        pdf.cell(35, 8, growth,  border=1)
        pdf.ln()
    pdf.ln(5)

    # Section 3
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Top Products", new_x="LMARGIN", new_y="NEXT")
    products = [
        ("Cloud Storage Pro",   "$12.5M", "Enterprise"),
        ("Data Pipeline Suite", "$10.2M", "Enterprise"),
        ("Analytics Dashboard", "$8.8M",  "SMB"),
        ("API Gateway",         "$7.1M",  "Developer"),
        ("Security Shield",     "$5.9M",  "Enterprise"),
    ]
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(70, 8, "Product",  border=1)
    pdf.cell(40, 8, "Revenue",  border=1)
    pdf.cell(40, 8, "Segment",  border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 11)
    for product, revenue, segment in products:
        pdf.cell(70, 8, product,  border=1)
        pdf.cell(40, 8, revenue,  border=1)
        pdf.cell(40, 8, segment,  border=1)
        pdf.ln()
    pdf.ln(5)

    # Section 4
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Quarterly Performance", new_x="LMARGIN", new_y="NEXT")
    quarters = [
        ("Q1 2024", "$10.2M", "$2.1M", "20.6%"),
        ("Q2 2024", "$11.5M", "$2.4M", "20.9%"),
        ("Q3 2024", "$12.8M", "$2.8M", "21.9%"),
        ("Q4 2024", "$14.0M", "$3.0M", "21.4%"),
    ]
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(40, 8, "Quarter", border=1)
    pdf.cell(40, 8, "Revenue", border=1)
    pdf.cell(40, 8, "Profit",  border=1)
    pdf.cell(40, 8, "Margin",  border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 11)
    for quarter, revenue, profit, margin in quarters:
        pdf.cell(40, 8, quarter, border=1)
        pdf.cell(40, 8, revenue, border=1)
        pdf.cell(40, 8, profit,  border=1)
        pdf.cell(40, 8, margin,  border=1)
        pdf.ln()
    pdf.ln(5)

    # Section 5
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "5. Key Challenges & Risks", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8,
        "1. Increased competition in the cloud storage segment led to 5% price reduction.\n"
        "2. Supply chain disruptions in Asia Pacific impacted hardware delivery timelines.\n"
        "3. Currency fluctuations reduced European revenue by approximately $0.8 million.\n"
        "4. Customer churn rate increased from 8% to 11% in the SMB segment.\n"
        "5. Hiring challenges delayed product launches by an average of 6 weeks."
    )
    pdf.ln(5)

    # Section 6
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "6. 2025 Targets", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8,
        "Revenue target for 2025 is $58 million representing 20% growth over 2024. "
        "Key focus areas include expanding the Asia Pacific market, launching 3 new products "
        "in the developer segment, and reducing customer churn to below 7%. "
        "Marketing budget has been increased by 25% to support these goals."
    )

    os.makedirs("data", exist_ok=True)
    output_path = "data/sample.pdf"
    pdf.output(output_path)
    print(f"PDF created: {output_path}")

if __name__ == "__main__":
    create_sales_report_pdf()