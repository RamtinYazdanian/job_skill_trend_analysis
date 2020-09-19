import pandas as pd
import numpy as np
from utilities.common_utils import remove_nones

def company_name_partial_match(s, s_ref):
    return s_ref.lower().find(s.lower()+' ') == 0 or s.lower()==s_ref.lower()

def list_partial_match(s, l):
    return [x for x in l if company_name_partial_match(s,x)]

def merge_with_main_table(df, main_df, merging_col, revenue_col, employee_col):
    df = df[remove_nones([merging_col, revenue_col, employee_col])]
    company_names_new = df[merging_col].values.tolist()
    list_of_companies_null = main_df.loc[(pd.isna(main_df['AnnualRevenue']))
                                         | (pd.isna(main_df['Employees'])), 'Company'].values.tolist()
    matched_companies = {x:list_partial_match(x, list_of_companies_null)
                 for x in company_names_new if len(list_partial_match(x, list_of_companies_null)) > 0}
    new_df = df.loc[df[merging_col].apply(lambda x: x in matched_companies)]
    new_df['Company'] = new_df[merging_col].apply(lambda x: matched_companies[x])
    new_df = new_df.explode('Company')
    results_df = pd.merge(main_df, new_df, on='Company', how='outer')

    if revenue_col is not None:
        results_df['AnnualRevenue'] = results_df.apply(lambda x: x['AnnualRevenue']
                                            if not pd.isna(x['AnnualRevenue']) else str(x[revenue_col])+'mil USD'
                                                       if not pd.isna(x[revenue_col]) else np.nan,
                                                                                                axis=1)
    if employee_col is not None:
        results_df['Employees'] = results_df.apply(lambda x: x['Employees']
                                        if not pd.isna(x['Employees']) else x[employee_col], axis=1)

    return results_df.drop(columns=remove_nones([merging_col, revenue_col, employee_col]))
