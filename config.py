#
# Stock Configuration

COMPANIES = {
    "Reliance Industries": ("Reliance", "RELIANCE.NS"),
    "Tata Consultancy Services": ("TCS", "TCS.NS"),
    "Infosys": ("Infosys", "INFY.NS"),
    "HDFC Bank": ("HDFC_Bank", "HDFCBANK.NS"),
    "ICICI Bank": ("ICICI_Bank", "ICICIBANK.NS"),
    "State Bank of India": ("SBI", "SBIN.NS"),
    "Wipro": ("Wipro", "WIPRO.NS"),
    "HCL Technologies": ("HCLTech", "HCLTECH.NS")
}

# Default company for existing standalone scripts
COMPANY_NAME, TICKER = COMPANIES["Reliance Industries"]

# Historical data start date
START_DATE = "2020-01-01"

# None means download through the latest available date
END_DATE = None
