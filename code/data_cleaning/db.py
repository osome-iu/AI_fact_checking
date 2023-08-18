"""
The columns used for each database table + a convenience function.

Author: Matthew DeVerna
"""
import os
import glob


def find_file(directory, matching_str):
    """
    Find a file in a directory using a wildcard pattern.

    Parameters:
    -----------
    - directory (str): Directory to search for files.
    - matching_str (str): Wildcard pattern to match files.

    Returns:
    - matched_file: full path to matched file
    """
    # Create the full pathname of the directory
    directory_path = os.path.abspath(directory)

    # Find all files matching the wildcard pattern in the directory
    matched_files = glob.glob(os.path.join(directory_path, matching_str))

    # We should only ever match one file
    if len(matched_files) != 1:
        raise Exception(
            f"Found {len(matched_files)} files matching the wildcard pattern: {matching_str}"
        )

    return matched_files[0]


participant_group_condition = [
    "ResponseId",  # Unique ID for each participant
    "Group",  # Options: ['Share', 'Belief']
    "Condition",  # Options: ['Control', 'Treatment II', 'Treatment I']]
]

meta_data = [
    "StartDate",  # Time survey started: E.g.: 2023-02-05 13:52:00
    "EndDate",  # Time survey ended: E.g.: 2023-02-05 14:38:46
    "Status",  # Options: ['Survey Preview', 'IP Address', 'Spam']
    "Progress",  # Percent complete (numpy.int64)
    "Duration (in seconds)",  # Total survey time (numpy.int64)
    "Finished",  # Did they finish (bool)
    "RecordedDate",  # Date survey was recorded: E.g.: 2023-02-05 14:38:47
    "term",  # Reason for termination (str). NaN if not applicable
    "residence",  # Do you reside in the US?
    "consent",  # Will you consent to participate in the survey?
    "self-vouching",  # Will you agree to try on the survey?
    "DistributionChannel",  # Options: ["preview", "anonymous"]
    "gc",  # "Good complete" codes
]

demographics = [
    "year_birth",  # Year of birth (numpy.float64)
    "gender",  # Selected gender (str) Options: ['Male', nan, 'Female', 'Prefer not to answer', 'Other']
    "gender_9_TEXT",  # Input if gender selected is "Other"
    "race",  # Concatenated string of all race items chosen
    "race_7_TEXT",  # Input if "Other" is slected
    "income",  # Selected income. Options: ['Less than $10,000', ..., '$150,000 or more']
    "state",  # Selected state of residence
    "edu",  # Selected education level. Options: ['Less than high school', ..., 'Doctoral degree']
]

# Questions related to how often news is accessed. See survey for questions.
# Answers options are:
#  - 'Never'
#  - 'About once every few months'
#  - 'About once a month'
#  - 'About once a week'
#  - 'A few times a week'
#  - 'About once a day'
#  - 'A few times day'
news_access = [
    "news_access_1",
    "news_access_2",
    "news_access_3",
    "news_access_4",
]

# Questions related to party identification. See survey for questions.
party_ids = [
    "party_id",  # Options: ['Democrat', 'Republican', 'Independent', 'No preference', 'Other', "Don't know"]
    "party_id_4_TEXT",  # Text input if "party_id" is "Other"
    "party_strength",  # If Democrat or Republican is chosen. Options: ['Strong', 'Somewhat strong']
    "Ind_leaning",  # If "Independent" is chosen. Options: ['Democratic Party', 'Republican Party', 'Neither', "Don't know"]
]

platform_usage = [
    "sm_platforms",  # String concatenation of all platforms participant reported as using
]

# Questions related to attitudes about AI.
# Ref: https://link.springer.com/article/10.1007/s13218-020-00689-0/tables/1
# Answers options are:
#  - 'Strongly disagree'
#  - 'Disagree'
#  - 'Somewhat disagree'
#  - 'Neither agree nor disagree'
#  - 'Somewhat agree'
#  - 'Agree'
#  - 'Strongly agree'
ai_attitudes = [
    "att_ai_1",
    "att_ai_2",
    "att_ai_3",
    "att_ai_4",
]

# Same as above, but after the experiment.
ai_attitudes_post = [
    "ai_att_post_1",
    "ai_att_post_2",
    "ai_att_post_3",
    "ai_att_post_4",
]

fact_checking_freq = ["FC_freq"]  # Likert about how often participant fact-checks news

# Did you search the internet for more information about the headlines you were asked about?
internet_search_q = ["search_internet"]

# Have you ever used AI powered tools like ChatGPT before? (Yes/No)
past_cgpt_usage = ["past_chatgpt"]

# If a subject answers "Yes" to the above, they then answer the below questions.
cgpt_pretest = [
    "chatgpt_freq",  # Frequency likert: In the past 30 days, how often have you used AI-powered tools like ChatGPT?
    "past_chatgpt_fc",  # Have you ever used AI powered tools like ChatGPT to fact-check before? (Yes/No)
    "pre_eval_1",  # Agreement likert: ChatGPT performs really well wehn fact-checking news reports
    "pre_eval_2",  # Agreement likert: ChatGPT outperforms existing fact-checking services
    "pre_eval_3",  # Agreement likert: Fact-checking answers provided by ChatGPT can change my mind
    "pre_eval_4",  # Agreement likert: Fact-checking answers provided by ChatGPT are objective
    "pre_eval_5",  # Agreement likert: Fact-checking answers provided by ChatGPT are trustworthy
    "pre_eval_6",  # Agreement likert: Fact-checking answers provided by ChatGPT are informative
    "pre_future_use_1",  # Agreement likert: I would like to use ChatGPT to verify information in the future on a regular basis
    "pre_future_use_2",  # Agreement likert: I hope SOCIAL MEDIA (e.g., Facebook, Twitter) incorporate ChatGPT fact-checking in their service
    "pre_future_use_3",  # Agreement likert: I hope SEARCH ENGINES (e.g., Google, Bing) incorporate ChatGPT fact-checking in their service
    "pre_future_use_4",  # Agreement likert: I hope NEWS AGGREGATION APPES (e.g., Apple News, Flipboard) incorporate ChatGPT fact-checking in their service
    "pre_future_use_5",  # Agreement likert: I will recommend ChatGPT fact-checking services to other people
]

cgpt_post_test = [
    "post_eval_1",  # Same as pre_eval_x questions, but after the experiment
    "post_eval_2",
    "post_eval_3",
    "post_eval_4",
    "post_eval_5",
    "post_eval_6",
    "post_future_use_1",  # Same as pre_future_use_X questions, but after the experiment
    "post_future_use_2",
    "post_future_use_3",
    "post_future_use_4",
    "post_future_use_5",
]

voting_post_test = [
    "vote_intent",  # Did you vote in the 2020 Presidential election? (Yes/No)
    "vote_2020_president",  # Who did you vote for in the 2020 Presidential election? (Trump/Biden/Other)
    "Midterm",  # Did you vote in the 2022 midterm election? (Yes/No)
]

# Feeling thermometer questions
affective_pol_post_test = [
    "Aff_polar_4",  # Republican voters
    "Aff_polar_5",  # Democrat voters
    "Aff_polar_6",  # Republican party
    "Aff_polar_7",  # Democrat party
]

# Personal identifiers and useless columns that we don't want
columns_to_drop = [
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "LocationLatitude",
    "LocationLongitude",
    "IPAddress",
    "ExternalReference",
    "UserLanguage",
    "Q_RecaptchaScore",
    "Q_RelevantIDDuplicate",
    "Q_RelevantIDDuplicateScore",
    "Q_RelevantIDFraudScore",
    "Q_RelevantIDLastStartDate",
    "SC0",
    "opp",
    "Q_BallotBoxStuffing",
    "Q_TotalDuration",
    "Q_CHL",
    "transaction_id",
    "SVID",
    "PS",
    "rid",
    "RISN",
    "V",
    "LS",
    "race.1",
    "version",
    "FL_79_DO",
    "Posttest_DO",
    "party_id_DO",
]
