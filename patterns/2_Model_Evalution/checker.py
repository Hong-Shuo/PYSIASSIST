import src.src as src

test_scripts = [r"2_Code_to_be_Checked.py"]

isadd, isfind = src.main(r"2_subchecker_1.json",test_scripts)

print(isadd, isfind)

if (not isfind ):
    isadd_2, isfind_2 = src.main(r"2_subchecker_2.json", test_scripts)
    print(isfind_2)
if (not isfind and not isfind_2):
    print("Warning: No operation was detected in your code to switch the model to evaluation mode. Please be mindful of this.")