from weasyprint import HTML
import os
# Use a real image path from your project
image_path = r"C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\static\person1_virus_11.jpeg"
html = f"""
<!DOCTYPE html>
<html>
<body>
    <h1>Test Report</h1>
    <img src="file://{image_path}">
</body>
</html>
"""
HTML(string=html).write_pdf("test.pdf")