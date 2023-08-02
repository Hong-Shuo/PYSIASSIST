import src.src as src

test_scripts = [r"1_test_code.py"]

isadd, isfind = src.main(r"1_subchecker.json",test_scripts)

print(isadd, isfind)