import sys
from fpdf import FPDF

class PDF(FPDF):
    def add_text_file(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                self.set_font('Arial', size = 8)
                self.cell(200, 5, txt = line, ln = True)

if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    print("No input file provided")
    sys.exit(1)

pdf = PDF()
pdf.add_page()
pdf.add_text_file(input_file)
pdf.output(input_file.replace('.txt', '.pdf'))
