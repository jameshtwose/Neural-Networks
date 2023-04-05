# %%
import pdfquery
# %%
# read the pdf file
filename = "jt-static-cv"
pdf = pdfquery.PDFQuery(f"data/{filename}.pdf")
pdf.load()

# %%
# convert the pdf to xml
pdf.tree.write(f"data/{filename}.xml", pretty_print=True)
pdf
# %%
# extract the text from the pdf
# Use CSS-like selectors to locate the elements
text_elements = pdf.pq('LTTextLineHorizontal')

# Extract the text from the elements
text = [t.text for t in text_elements if len(t.text) > 1]

print(text)
# %%
text_elements = pdf.pq('LTTextBoxHorizontal')

# Extract the text from the elements
text = [t.text for t in text_elements if len(t.text) > 1]

print(text)
# %%
