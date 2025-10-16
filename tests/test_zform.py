"""
This is a quick test for the zform function
using the penguins seaoborn dataset

zform is part of the zlab library by Nicolò Zorzetto

License
-------
GPL v3
"""
import os
import sys

try:
    # Works when running as a file
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    # Fallback for interactive environments (no __file__)
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from zlab import zform

def try_zform_penguins():
    try:
        import seaborn as sbn
    except ImportError:
        raise ImportError(
            "You do not have seaborn installed"
        )
    while True:
        print("Which dataset do you want to use?")
        print("\t 1 - small penguins dataset (original 333 rows)")
        print("\t 2 - big penguins dataset (scaled to 333'000 rows)")
        which_data = input()
        if which_data == "1":
            df = sbn.load_dataset("penguins").dropna()
            break
        if which_data == "2":
            df = sbn.load_dataset("penguins").dropna()
            try:
                import pandas as pd
            except ImportError as e:
                raise ImprtError(
                    "You do not have pandas installed."
                       "Please install all requirements via "
                      "'pip install-r requirements.txt' "
                )
            df = pd.concat([df] * 1000, ignore_index=True)
            break
        else:
            print("Only input 1 for the small dataset, or 2 for the big one.")
    df_out, forms = zform(
        df,
        group_col="species",
        return_results=True,
	    min_obs = 10,
	    apply = True,
	    naming = "standard",
        export_csv="./forms.csv",
        n_jobs=-1
    )

    while True:
        print_question = input("Would you like to check the results? (y/n) ")
        if print_question.lower()[0] == "y":
            print("Which results would you like to check?")
            print("\t 1 - forms dataframe")
            print("\t 2 - forms best metric")
            print("\t 3 - original and transformed columns after application to original df (standard naming)")
            which_print = input("")
            if which_print[0] == '1':
                print(forms.head(10))
                print("To view different naming options use help(zform) or help(zform_apply)")
            elif which_print[0] == '2':
                print(list(forms.columns)[4])
                print(forms[list(forms.columns)[4]].head(10))
            elif which_print[0] == '3':
                print(df_out.head(10))
            else:
                print("Please input a valid selection.")
        elif print_question.lower()[0] == "n":
            break
        else:
            print("Please input just y or n.")


if __name__ == "__main__":
    while True:
        run_test_input = input("Would you like to run the penguins test? (y/n) ")
        if run_test_input.lower()[0] == "y":
            try_zform_penguins()
            break
        if run_test_input.lower()[0] == "n":
            print("Not running the test.")
            break
        else:
            print("Please input y or n")
