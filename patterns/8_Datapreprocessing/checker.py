import src.src as src

test_scripts = [r"8_test_code.py"]

isadd, isfind = src.main(r"8_subchecker.json",test_scripts)

print(isadd, isfind)