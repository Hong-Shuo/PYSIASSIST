import src.src as src

test_scripts = [r"9_test_code.py"]

isadd, isfind = src.main(r"9_subchecker.json",test_scripts)

print(isadd, isfind)