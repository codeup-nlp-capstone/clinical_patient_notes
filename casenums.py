import pandas as pd

class UniqueDataFrames:
    """ This class merges to dataframes and then creates a dictionary where the key is the unique column entry and the value is the dataframe relating to that unique column entry
    """

    def __init__(self, notes_file, feature_file):
        """__init__ initialize class with filename

        Args:
            notes_file (string): filenname of first df
            feature_file (string): filename of second df
        """

        self.patient_notes = pd.read_csv(notes_file)
        self.features = pd.read_csv(feature_file)

    def merge_dfs(self, notes_col="case_num", features_col1="feature_num", features_col2="feature_text", notes_featues_col="features"):
        """merge_dfs merge the patient notes and features dfs

        Args:
            notes_col (str, optional): column in notes on which to merge and the first column in new df created from dictionary d. Defaults to "case_num".
            features_col1 (str, optional): key for dictionary. Defaults to "feature_num".
            features_col2 (str, optional): value of dictionary. Defaults to "feature_text".
            notes_featues_col (str, optional): name of new column created by the dictionary d. Defaults to "features".

        Returns:
            pandas dataframe: the new merged dataframe of patient notes and features
        """

        d = {}
        for i in self.features[notes_col].unique():
            d[i] = {a: self.features[features_col2][self.features[features_col1] == a].to_string()[5:].strip() for a in self.features[features_col1][self.features[notes_col] == i]}

        features_dict = pd.DataFrame(data=[d.keys(), d.values()]).T
        features_dict.columns = [notes_col, notes_featues_col]

        return self.patient_notes.merge(right=features_dict, left_on="case_num", right_on="case_num")

    def each_dataframe(self, notes_features, notes_name="case_num"):
        """each_dataframe creates a dicationary of separate dataframes based off of unqiue case numbers

        Args:
            notes_features (pandas dataframe): the dataframe used to create the new dataframe
            notes_name (str, optional): column to use to get unqiue keys. Defaults to "case_num".

        Returns:
            dict: dictionary where the key is a case number and the value is the dataframe specific to that case number
        """

        return {case_num: notes_features[notes_features[notes_name] == case_num] for case_num in notes_features[notes_name].unique()}