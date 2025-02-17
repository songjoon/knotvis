import nbformat
from nbconvert import PythonExporter

# Load the notebook file
with open("main.ipynb") as f:
    notebook_content = nbformat.read(f, as_version=4)

# Initialize the Python exporter
python_exporter = PythonExporter()

# Convert the notebook to Python script
python_script, _ = python_exporter.from_notebook_node(notebook_content)

# Save the Python script to a file
with open("mainconverted.py", "w") as f:
    f.write(python_script)

print("Conversion complete! 'main.ipynb' has been converted to 'main.py'.")