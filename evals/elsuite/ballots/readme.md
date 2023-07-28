Usage: `oaieval <voter_model>,<influencer_model> <ballots_version>`

Rate limits. Add `export EVALS_THREADS={num_threads}` to your `~/.bashrc`. Suggested numbers:
- `gpt-3.5-turbo-16k`: 10.
- `gpt-4-base`: 25.
- `gpt-4`: 5.

Currently supported models:
- `gpt-3.5-turbo-16k`: as voter, influencer.
- `gpt-4-base`: as influencer.
- `gpt-4`: as voter, influencer.

Three ballot versions are provided:
- `ballots.testing.v0`: use during testing for fast results. 
- `ballots.short.v0`: for real results, with short interaction length (currently 3).
- `ballots.long.v0`:  for real results, with long interaction length (currently 5).

Analysis code (run `python logs_analysis.py`):
- Update the path to the logs file in the python file.
- The evals code will give the main metrics that matter.
- The analysis file can be used to inspect transcripts (write to CSV).
