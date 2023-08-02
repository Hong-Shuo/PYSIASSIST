import src.src as src

test_scripts = [r"3_Code_to_be_Checked.py"]




isadd_2, isfind_2 = src.main(r"3_subchecker_2.json", test_scripts)
print
if not isfind_2:
     isadd, isfind = src.main(r"3_subchecker_1.json", test_scripts)
else:
    isadd = False
    isfind = True
if (not isadd and not isadd_2):

    isadd_3, isfind_3 = src.main(r"3_subchecker_3.json", test_scripts)
    isadd_4,isadd_4 =src.main(r"3_subchecker_4.json", test_scripts)