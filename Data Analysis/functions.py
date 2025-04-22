import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


def parse_xml_file(file_path):
    """
    Parse an XML file using BeautifulSoup.

    Parameters:
        file_path (str): Path to the XML file.

    Returns:
        BeautifulSoup: Parsed XML data using the 'xml' parser.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return BeautifulSoup(data, "xml")


def flatten_list(nested_list):
    """
    Recursively flatten a nested list.

    Parameters:
        nested_list (list): A list which may contain nested lists.

    Returns:
        list: A single flattened list containing all elements.
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def extract_nested_tags(element, prefix=''):
    """
    Recursively extract tags and attributes from an XML element into a dictionary.

    Parameters:
        element (Tag): A BeautifulSoup Tag object.
        prefix (str): A prefix string to prepend to keys (used during recursion).

    Returns:
        dict: A dictionary with tag text and attributes, where keys represent the nested structure.
    """
    data = {}

    # Include the element's text if present and non-empty.
    if element.string and element.string.strip():
        data[prefix.rstrip('_')] = element.string.strip()

    # Include element attributes.
    for attr, value in element.attrs.items():
        data[f"{prefix}{attr}"] = value

    # Recursively extract data from child elements.
    for child in element.find_all(recursive=False):
        child_prefix = f"{prefix}{child.name}_"
        child_data = extract_nested_tags(child, child_prefix)
        for key, value in child_data.items():
            if key in data:
                # Combine multiple values into a list.
                if isinstance(data[key], list):
                    data[key].append(value)
                else:
                    data[key] = [data[key], value]
            else:
                data[key] = value

    return data


def xml_to_dataframe(soup_data):
    """
    Convert XML data (parsed with BeautifulSoup) into a Pandas DataFrame,
    while also identifying columns that contain nested lists.

    Parameters:
        soup_data (BeautifulSoup): Parsed XML data containing `<repository>` elements.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with cleaned and simplified column names.
            - list: List of column names that contain nested lists.
    """
    repo_list = []
    for repo in soup_data.find_all('repository'):
        repo_data = extract_nested_tags(repo)
        repo_list.append(repo_data)

    df = pd.DataFrame(repo_list)

    # Identify columns that contain lists.
    list_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]

    # Clean and standardize column names.
    original_columns = df.columns
    cleaned_columns = [
        col.lower()
           .replace('.', '_')
           .replace('repository', 'repo')
           .replace('institution_', '')
           .replace('metadatastandard_', '')
           .replace('databaseaccess_', '')
           .replace('databaselicense_', '')
           .replace('datauploadlicense_', '')
           .replace('subject_', '')
           .replace('contenttype_', '')
           .replace('datalicense_', '')
           .replace('metadatastandardname_', '')
           .replace('dataaccess_', '')
           .replace('dataLicense_', '')
           .replace('dataupload_', '')
           .replace('contentType_', '')
           .replace('policy_', '')
           .replace('syndication_', '')
           .replace('additionalname', 'alt_name')
           .replace('responsibility', 'resp')
           .replace('language', 'lang')
           .replace('name', 'name')
           .replace('url', 'url')
           .strip('_')
        for col in original_columns
    ]
    df.columns = cleaned_columns

    return df, list_columns


def count_nonupdated_entries(dataframe):
    """
    Count the number of entries where the entry date equals the last update date.

    Parameters:
        dataframe (pd.DataFrame): DataFrame with 'entrydate' and 'lastupdate' columns.

    Returns:
        int: Count of entries that have never been updated.
    """
    count = (dataframe['entrydate'] == dataframe['lastupdate']).sum()
    print(f"Number of entries that are never updated: {count}")
    return count


def plot_update_history(dataframe, frequency='M', figure_size=(10, 6)):
    """
    Plot the number of last update entries over time, aggregated by a specified frequency.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing a 'lastupdate' column.
        frequency (str): Resampling frequency (e.g., 'M' for monthly, 'W' for weekly).
        figure_size (tuple): Size of the plot.

    Returns:
        None
    """
    dataframe['lastupdate'] = pd.to_datetime(dataframe['lastupdate'])
    resampled_counts = dataframe.set_index('lastupdate').resample(frequency).size()

    plt.figure(figsize=figure_size)
    plt.plot(resampled_counts.index, resampled_counts.values,
             linestyle='-', marker='o', label='Last Update Counts')
    plt.title('Entries Last Updated Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Updates')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def extract_dra_columns(dataframe):
    """
    Select a predefined set of columns from the DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing only the selected columns.
    """
    selected_columns = [
        're3data_orgidentifier', 'reponame', 'repourl', 'institutioncountry', 'repolang',
        'institutionname', 'repocontact', 'description', 'subject', 'subjectscheme', 'repoidentifier',
        'institutionidentifier', 'pidsystem', 'aidsystem', 'api', 'api_apitype', 'metadatastandardname',
        'metadatastandardscheme', 'metadatastandardurl', 'qualitymanagement', 'datauploadtype',
        'datauploadrestriction', 'datauploadlicensename', 'datauploadlicenseurl', 'dataaccessrestriction',
        'dataaccesstype', 'databaseaccessrestriction', 'databaseaccesstype', 'datalicensename',
        'datalicenseurl', 'certificate'
    ]
    return dataframe[selected_columns]


def generate_column_distribution_stats(dataframe, columns_list=None, output_file='columns_distribution.xlsx', visualize=True):
    """
    Generate distribution statistics for specified columns, write the statistics to an Excel file,
    and optionally visualize the top 20 values for each column.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing the data.
        columns_list (list, optional): List of column names to process. If None, a default list is used.
        output_file (str): Filename (or path) for the output Excel file.
        visualize (bool): If True, display a bar chart for each column's distribution.

    Returns:
        None
    """
    if columns_list is None:
        columns_list = [
            'institutioncountry', 'repolang', 'subject', 'subjectscheme', 'api_apitype',
            'metadatastandardname', 'metadatastandardscheme', 'qualitymanagement',
            'datauploadtype', 'datauploadrestriction', 'datauploadlicensename',
            'datalicensename', 'dataaccessrestriction', 'dataaccesstype',
            'databaseaccessrestriction', 'databaseaccesstype', 'certificate', 'pidsystem', 'aidsystem'
        ]

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for column in columns_list:
            if column not in dataframe.columns:
                print(f"Column {column} does not exist in the DataFrame.")
                continue
            print(f'Column {column} has been selected.')


            # Handle missing values and flatten lists if needed.
            column_data = dataframe[column].dropna().tolist()
            all_values = []
            for value in column_data:
                if isinstance(value, list):
                    all_values.extend(flatten_list(value))
                else:
                    all_values.append(value)

            value_counts = Counter(all_values)
            total_count = sum(value_counts.values())

            if total_count == 0:
                print(f"No valid data available for column {column} after processing.")
            else:
                stats_df = pd.DataFrame(
                    [(val, count, (count / total_count) * 100) for val, count in value_counts.items()],
                    columns=['Value', 'Count', 'Percentage']
                ).sort_values(by='Count', ascending=False)

                # Write to Excel
                sheet_name = column[:31]
                stats_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Statistics for {column} written to Excel sheet.")

                # Plotting
                if visualize:
                    plt.figure(figsize=(14, 8))
                    stats_df_top = stats_df.head(20).reset_index(drop=True)

                    # Create the bar plot with seaborn for better aesthetics
                    sns.barplot(x='Value', y='Count', data=stats_df_top, palette='husl')

                    # Add labels on top of bars
                    for i, row in stats_df_top.iterrows():
                        plt.text(i, row['Count'] + max(stats_df_top['Count']) * 0.01,
                                 f"{row['Count']}\n({row['Percentage']:.1f}%)",
                                 ha='center', va='bottom', fontsize=10)

                    # Labeling with larger fonts
                    plt.xlabel('Value', fontsize=14, fontweight='bold')
                    plt.ylabel('Count', fontsize=14, fontweight='bold')
                    plt.title(f'Distribution of {column}', fontsize=16, fontweight='bold')

                    # Ensure x-axis labels are readable and aligned
                    plt.xticks(ticks=range(len(stats_df_top['Value'])), labels=stats_df_top['Value'], rotation=45,
                               fontsize=12, ha='right')
                    plt.yticks(fontsize=12)

                    plt.tight_layout()
                    plt.show()

    print(f"All statistics have been saved to {output_file}.")


def save_complete_dataset(dataframe, output_file='most_complete_repo_list.csv'):
    """
    Filter the DataFrame by dropping rows with any missing values and save the result to a CSV file.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        output_file (str): Filename (or path) for the output CSV file.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only complete rows.
    """
    complete_df = dataframe.dropna(how='any')
    complete_df.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")
    return complete_df


def save_top_complete_rows(dataframe, top_n=30, output_file='top_30_most_complete_rows.csv'):
    """
    Select the top rows with the highest number of non-missing values,
    save these rows to a CSV file, and return the resulting DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        top_n (int): The number of top rows to select.
        output_file (str): Filename (or path) for the output CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the top `top_n` most complete rows.
    """
    df_copy = dataframe.copy()
    df_copy['non_nan_count'] = df_copy.notna().sum(axis=1)
    sorted_df = df_copy.sort_values(by='non_nan_count', ascending=False)
    top_complete_df = sorted_df.head(top_n)
    top_complete_df.to_csv(output_file, index=False)
    print(f"Top {top_n} most complete rows saved to {output_file}")
    return top_complete_df


def compute_completeness_report(dataframe):
    """
    Compute a completeness report for the DataFrame, based on the percentage of non-missing values per column.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to evaluate.

    Returns:
        pd.DataFrame: A report with columns 'completeness (%)' and 'missing_count' per column.
    """
    total_rows = len(dataframe)
    completeness = dataframe.notnull().sum() / total_rows * 100
    missing = dataframe.isnull().sum()

    report = pd.DataFrame({
        'completeness (%)': completeness,
        'missing_count': missing
    })
    return report


def compare_two_completeness(dataframe_all, dataframe_filtered, output_file = None):
    """
    Compare the completeness between two DataFrames and return a combined report.

    Parameters:
        dataframe_all (pd.DataFrame): The full DataFrame.
        dataframe_filtered (pd.DataFrame): A subset of the full DataFrame (e.g., filtered by institution).
        output_file (str): Filename (or path) for the output Excel file.

    Returns:
        pd.DataFrame: A report showing completeness percentages, missing counts, and the difference.
    """
    report_all = compute_completeness_report(dataframe_all)
    report_filtered = compute_completeness_report(dataframe_filtered)
    combined = report_all.join(report_filtered, lsuffix='_all', rsuffix='_filtered')
    combined['difference (%)'] = combined['completeness (%)_filtered'] - combined['completeness (%)_all']
    if output_file:
        combined.to_excel(output_file, index=True)
        print(f"Completeness comparison report saved to {output_file}")
    return combined


def compare_multiple_completeness(dataframes_dict, baseline=None):
    """
    Compare the completeness of multiple DataFrames and return a combined report.

    Parameters:
        dataframes_dict (dict): A dictionary with keys as labels and values as DataFrames.
        baseline (str, optional): The key in dataframes_dict to use as a baseline for difference calculations.

    Returns:
        pd.DataFrame: A report combining completeness metrics and differences relative to the baseline.
    """
    reports = {}
    for label, df in dataframes_dict.items():
        report = compute_completeness_report(df)
        report = report.rename(columns={
            "completeness (%)": f"completeness (%)_{label}",
            "missing_count": f"missing_count_{label}"
        })
        reports[label] = report

    combined_report = pd.concat(reports.values(), axis=1)

    if baseline is not None and baseline in reports:
        base_col = f"completeness (%)_{baseline}"
        for label in reports.keys():
            if label == baseline:
                continue
            diff_col = f"difference (%)_{label}"
            combined_report[diff_col] = combined_report[f"completeness (%)_{label}"] - combined_report[base_col]

    return combined_report


