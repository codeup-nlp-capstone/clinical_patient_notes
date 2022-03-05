import pandas as pd

class UniqueDataFrames:

    def __init__(self, notes_file, feature_file):

        self.patient_notes = pd.read_csv(notes_file)
        self.features = pd.read_csv(feature_file)

    def merge_dfs(self, notes_col="case_num", features_col1="feature_num", features_col2="feature_text", notes_featues_col="features"):

        d = {}
        for i in self.features[notes_col].unique():
            d[i] = {a: self.features[features_col2][self.features[features_col1] == a].to_string()[5:].strip() for a in self.features[features_col1][self.features[notes_col] == i]}

        features_dict = pd.DataFrame(data=[d.keys(), d.values()]).T
        features_dict.columns = [notes_col, notes_featues_col]

        return self.patient_notes.merge(right=features_dict, left_on="case_num", right_on="case_num")

    def each_dataframe(self, notes_features, notes_name="case_num"):

        return {case_num: notes_features[notes_features[notes_name] == case_num] for case_num in notes_features[notes_name].unique()}