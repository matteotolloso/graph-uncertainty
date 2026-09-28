# Anonymity (double-blind submission)

The `AAAI` line of branches is shared as anonymous supplementary code (see the "anonimization" commit).
In every tracked file (code, configs, docs, notebooks, commit-ready outputs):

- no author names, usernames, e-mail addresses, institutions or acknowledgements;
- no W&B entity / team names, run URLs or sweep URLs — code must obtain the entity at runtime
  (`wandb.Api().default_entity`), never hard-code it;
- no absolute paths containing a home or user directory (`/raid/<user>/...`, `/home/<user>/...`);
  use paths relative to the repo or the `CGNN_*` environment variables;
- no hostnames, cluster names or IPs;
- notebooks: clear outputs that print paths or user names before committing.

Third-party attribution (papers, upstream repositories such as `graph-ebm`) is fine and required.
If you find identifying information in tracked files, report it to the user instead of rewriting history.
