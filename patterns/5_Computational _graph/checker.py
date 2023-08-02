import src.src as src

test_scripts = [r"5_test_code.py"]

isadd, isfind = src.main(r"5_subchecker.json",test_scripts)

print(isadd, isfind)