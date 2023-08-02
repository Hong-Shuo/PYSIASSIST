import src.src as src

test_scripts = [r"8_Code_to_be_Checked.py"]

isadd, isfind = src.main(r"8_subchecker.json",test_scripts)

print(isadd, isfind)