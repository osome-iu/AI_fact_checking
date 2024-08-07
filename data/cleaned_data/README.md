## Data description

### Description
The data in this folder represents the cleaned data utilized for data analysis.
Rows represent a participant's response to an individual headline.

### Columns:
- `Group`: participant group
    - Values: ['Belief', 'Share']
- `Condition`: participant experimental condition
    - Values: ['Control', 'Optional', 'Forced', 'Human-FC']
- `option_cond`: participant experimental condition which also indicates whether Optional condition participants opted in or out of viewing a fact check
    - Values: ['Control', 'Forced', 'Opt_out', 'Opt_in']
- `ResponseId`: Qualtric's unique identifier for each participant
- `qualtrics_question_num`: the headline question number. Note that these numbers **do not** reflect the order in which they were presented, which was randomized in the qualtrics system for each participant.
    - Values: integers from 1 to 40
- `exp_response`: the participants response for a given headline
    - Values: [True, False] (boolean)
- `gender`: participant's self-reported gender
    - Values: ['Female', 'Male', 'Other']
- `year_birth`: particpant's self-reported birth year. Participants younger than 18 years of age were automatically excluded. Note that participants in the 'Control', 'Optional', and 'Forced' conditions participated in the study in 2023, while those in the 'Human-FC' condition participated in 2024.
- `race`: particpant's self-reported race.
    - Values: Various strings — as participants were allowed to select as many options as they wanted. See the preregistration details for more information and the actual qualtrics survey. If more than one option was selected, they are concatenated with comma separators.
- `edu`: participant's self-reported highest level of education.
    - Values: ["Bachelor's degree in college (4-year)",  'Associate degree in college (2-year)' "Master's degree",  'Some college but no degree', 'Professional degree (JD, MD)', 'High school graduate (high school diploma or equivalent including GED)',  'Doctoral degree', 'Less than high school degree']
- `edu_recoded`: the above mapped to integers from 1 to 8.
- `party_id`: participant's self-report party identification. 
    - Values: ['Democrat', 'Republican', 'Independent', 'No preference', "Don't know", 'Other']
- `party_recoded`: the `party_id` column recoded such that independents who "leaned" democrat/republican were coded as Democrats/Republicans. See paper for details on how this was done.
    - Values: ['Dem', 'Rep', 'Ind']
- `AI_att_mean`: participants estimated attitude towards artificial intelligence. Calculated as the mean value of a slightly altered version of [this battery](https://doi.org/10.1007/s13218-020-00689-0). See the paper for details.
    - Values: range(1,7)
- `congruency`: whether the headline is congruent with that participant's partisian position.
    - Values: ['congruent', 'incongruent', None]
    - Note: `None` values are given to participants coded as independents
- `veracity`: whether the headline was True or False
    - Values: [True, False] (boolean)
- `headline_text`: the headline text
- `cgpt_response`: ChatGPT's response. See the paper for model and prompt details
- `ano_true_false_unsure`: whether ChatGPT judged the headline to be True, False, or Unsure. Three authors manually coded responses independently. See the paper for coding details.
    - Values: ['unsure' 'false' 'true']